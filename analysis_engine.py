import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import multiprocessing as mp
import time
import os
import io
import traceback
try:
    import cadquery as cq
except Exception:
    cq = None

def _detect_mill_axis(solids):
    all_mins, all_maxs = [], []
    for s in solids:
        bb = s.BoundingBox()
        all_mins.append([bb.xmin, bb.ymin, bb.zmin])
        all_maxs.append([bb.xmax, bb.ymax, bb.zmax])
    bbox_min = np.min(all_mins, axis=0)
    bbox_max = np.max(all_maxs, axis=0)
    spans    = bbox_max - bbox_min
    center   = (bbox_min + bbox_max) / 2
    axis_names = ['X', 'Y', 'Z']
    mill_axis_idx = int(np.argmin(spans))
    plane_map  = {0: "YZ", 1: "XZ", 2: "XY"}
    plane_name = plane_map[mill_axis_idx]
    plane_axes = tuple(i for i in range(3) if i != mill_axis_idx)
    return axis_names[mill_axis_idx], plane_name, plane_axes, center, bbox_min, bbox_max


def _get_section_coords(point, plane_axes):
    coords = [point.x, point.y, point.z]
    return coords[plane_axes[0]], coords[plane_axes[1]]


def _collect_section_points(solids, plane_name, section_offset, plane_axes, log_cb=None):
    import cadquery as cq
    all_pts = []
    for i, solid in enumerate(solids):
        if log_cb:
            log_cb("inf", f"  Sectioning solid {i+1}/{len(solids)}…")
        try:
            # Create the workplane with offset first, THEN add the solid, so the section cuts exactly at the offset plane.
            wp = cq.Workplane(plane_name).workplane(offset=section_offset).add(solid)
            section = wp.section()
            wires = section.wires().vals()
            for w in wires:
                n_pts = max(500, int(w.Length() / 1.0))
                for j in range(n_pts):
                    try:
                        pt = w.positionAt(j / n_pts, mode="length")
                        c1, c2 = _get_section_coords(pt, plane_axes)
                        all_pts.append((c1, c2))
                    except Exception:
                        pass
        except Exception as ex:
            if log_cb:
                log_cb("wrn", f"  Solid {i+1} skipped: {ex}")
    return np.array(all_pts) if all_pts else np.array([]).reshape(0, 2)


def _build_polar_profile(all_points, n_angular=16384):
    # Algebraic circle fit (Taubin method) to robustly find the geometric center (handling offsets and odd lifters).
    x, y = all_points[:, 0], all_points[:, 1]
    # Initial algebraic fit on all points
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x**2 + y**2)
    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    c1c, c2c = -p[0]/2, -p[1]/2
    
    # Refine fit using only the outermost points (the shell)
    r_est = np.sqrt((x - c1c)**2 + (y - c2c)**2)
    shell_mask = r_est > (r_est.max() * 0.98)
    if shell_mask.sum() > 10:
        xs, ys = x[shell_mask], y[shell_mask]
        A_ref = np.c_[xs, ys, np.ones_like(xs)]
        b_ref = -(xs**2 + ys**2)
        p_ref, _, _, _ = np.linalg.lstsq(A_ref, b_ref, rcond=None)
        c1c, c2c = -p_ref[0]/2, -p_ref[1]/2
        
    center_2d = (c1c, c2c)
    d1 = all_points[:, 0] - c1c
    d2 = all_points[:, 1] - c2c
    theta_raw = np.arctan2(d2, d1) % (2 * np.pi)
    r_raw = np.sqrt(d1**2 + d2**2)
    r_shell = r_raw.max()
    theta_bins = np.linspace(0, 2 * np.pi, n_angular + 1)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    r_inner = np.full(n_angular, np.nan)
    r_outer = np.full(n_angular, np.nan)
    for k in range(n_angular):
        mask = (theta_raw >= theta_bins[k]) & (theta_raw < theta_bins[k + 1])
        if mask.any():
            r_inner[k] = r_raw[mask].min()
            r_outer[k] = r_raw[mask].max()
    valid = ~np.isnan(r_inner)
    if not valid.all() and valid.any():
        fi = interp1d(theta_centers[valid], r_inner[valid], kind='linear', fill_value='extrapolate')
        r_inner = fi(theta_centers)
        valid_o = ~np.isnan(r_outer)
        if valid_o.any():
            fo = interp1d(theta_centers[valid_o], r_outer[valid_o], kind='linear', fill_value='extrapolate')
            r_outer = fo(theta_centers)
    h_protrusion = r_shell - r_inner
    thickness    = r_outer - r_inner
    return theta_centers, r_inner, r_outer, h_protrusion, thickness, r_shell, center_2d


def _detect_lifters(h_protrusion, n_angular):
    """Detect lifter peaks using a multi-parameter sweep.
    
    Uses an inter-lifter spacing estimate to set the minimum peak-to-peak
    distance, avoiding double-detection of the two edges of a flat top-land.
    The sweep picks the most frequently agreed-upon lifter count.
    """
    h_range = h_protrusion.max() - h_protrusion.min()
    h_extended = np.concatenate([h_protrusion, h_protrusion, h_protrusion])
    results = {}
    for prom_frac in [0.02, 0.05, 0.10, 0.15, 0.20]:
        prom = max(1.0, h_range * prom_frac)
        for dist_div in [50, 60, 70, 80, 100]:
            min_dist = max(3, n_angular // dist_div)
            pks_ext, _ = find_peaks(h_extended, prominence=prom, distance=min_dist)
            pks = pks_ext[(pks_ext >= n_angular) & (pks_ext < 2 * n_angular)] - n_angular
            pks = np.unique(pks)
            n = len(pks)
            if n > 0:
                results.setdefault(n, []).append((prom, min_dist, pks))
    if not results:
        pks_ext, _ = find_peaks(h_extended, prominence=1.0, distance=3)
        pks = pks_ext[(pks_ext >= n_angular) & (pks_ext < 2 * n_angular)] - n_angular
        return np.unique(pks)
    count_freq = {n: len(c) for n, c in results.items()}
    best_count = max(count_freq, key=count_freq.get)
    candidate_pks = np.sort(results[best_count][0][2])
    
    # ── Second pass: if each lifter appears to have a double-peak (flat top-land),
    #    merge pairs using the expected inter-lifter period as the guard distance.
    #    A pair is two consecutive peaks spaced < 25% of the expected period.
    if best_count > 1:
        expected_period_bins = n_angular / best_count
        merge_threshold = 0.25 * expected_period_bins
        spacings = np.diff(candidate_pks)
        if spacings.min() < merge_threshold:
            # Merge pairs: take the midpoint of each tight pair
            merged = []
            i = 0
            while i < len(candidate_pks):
                if i + 1 < len(candidate_pks) and (candidate_pks[i+1] - candidate_pks[i]) < merge_threshold:
                    merged.append(int(round((candidate_pks[i] + candidate_pks[i+1]) / 2)))
                    i += 2
                else:
                    merged.append(candidate_pks[i])
                    i += 1
            candidate_pks = np.array(merged)
    
    return candidate_pks


def _arc_length_sample(theta_centers, r_inner, h_protrusion, n_arc=1024):
    x_inner = r_inner * np.cos(theta_centers)
    y_inner = r_inner * np.sin(theta_centers)
    dx = np.diff(x_inner, append=x_inner[0])
    dy = np.diff(y_inner, append=y_inner[0])
    ds = np.sqrt(dx**2 + dy**2)
    arc_len = np.cumsum(ds)
    arc_len = np.insert(arc_len, 0, 0)[:-1]
    total_arc = arc_len[-1] + ds[-1]
    s_norm = arc_len / total_arc
    s_uniform = np.linspace(0, 1, n_arc, endpoint=False)
    h_arc = np.interp(s_uniform, s_norm, h_protrusion, period=1.0)
    return s_uniform, h_arc, total_arc


def _fourier_analysis(h_arc, n_harmonics=120):
    H = np.fft.rfft(h_arc)
    mag = np.abs(H) / len(h_arc)
    mag[1:] *= 2
    n_show = min(n_harmonics, len(mag))
    dominant_k = int(np.argmax(mag[1:n_show])) + 1
    return mag, dominant_k


# ─────────────────────────────────────────────────────────────────────────────
# ML feature extraction
# ─────────────────────────────────────────────────────────────────────────────




def _true_period_via_autocorr(h, min_period_deg=1.0):
    n = len(h)
    h_c = h - h.mean()
    F  = np.fft.fft(h_c)
    ac = np.fft.ifft(F * np.conj(F)).real
    ac = ac / ac[0]
    min_lag = max(int(min_period_deg / 360.0 * n), 1)
    search  = ac[min_lag: n // 2]
    ac_best = search.max()
    pks, _  = find_peaks(search, height=0.995 * ac_best)
    pks     = pks + min_lag
    if len(pks) == 0:
        pks = np.array([int(np.argmax(search)) + min_lag])
    return pks[0] / n * 360.0, pks[0]


def _extract_unit_shape_vector(theta_deg, h, period_deg, n_samples):
    mask = theta_deg < (theta_deg.min() + period_deg)
    th, hh = theta_deg[mask], h[mask]
    s_local   = (th - th.min()) / period_deg
    s_uniform = np.linspace(0, 1, n_samples, endpoint=False)
    return np.interp(s_uniform, s_local, hh, period=1.0)


def _fourier_magnitude(h, n_harmonics):
    H   = np.fft.rfft(h)
    mag = np.abs(H) / len(h)
    mag[1:] *= 2
    return mag[:n_harmonics]


def _cluster_peaks_by_height(peak_heights):
    """Identify the PRIMARY (tallest) cluster of lifter peaks.

    Returns a boolean mask where True = primary (tallest) group.

    Uses the 90th-percentile height as reference (robust to one or two freak
    boundary-artefact spikes) then selects all peaks within 20 pct of that
    reference as primary.  Progressive threshold relaxation ensures at least
    2 primary peaks are always returned when possible.
    """
    if len(peak_heights) == 0:
        return np.ones(0, dtype=bool)
    # 90th percentile avoids freak spikes (e.g. wrap-around boundary artefact)
    h_ref = np.percentile(peak_heights, 90)
    for thr in [0.80, 0.75, 0.70, 0.60, 0.50]:
        primary = (peak_heights >= thr * h_ref) & (peak_heights <= 1.20 * h_ref)
        if primary.sum() >= 2:
            return primary
    # Last fallback: anything in the upper half
    return peak_heights >= 0.50 * h_ref
def get_true_valley(h_array, start_idx, end_bound, direction):
    """Walk down the profile from a peak to find its true local base.
    Stops when the profile stops descending (flattens or rises by >1mm).
    This correctly identifies plateaus vs deep grooves."""
    n = len(h_array)
    curr = start_idx
    valley = start_idx
    min_h = h_array[start_idx]
    
    for _ in range(n // 2):
        nxt = (curr + direction) % n
        if nxt == end_bound:
            break
        if h_array[nxt] < min_h:
            min_h = h_array[nxt]
            valley = nxt
        elif h_array[nxt] > min_h + 1.0:  # 1mm tolerance for flattening out
            break
        curr = nxt
    return valley

def cluster_lifter_patterns(h_protrusion, peaks):
    """
    Groups lifters into Tall, Medium, Short, Build-up based on their peak heights.
    """
    if len(peaks) == 0:
        return [], "None", np.array([]), []
        
    peak_h = h_protrusion[peaks]
    max_h = peak_h.max()
    min_h = h_protrusion.min()
    amp = max_h - min_h
    
    is_buildup = (peak_h - min_h) < 0.20 * amp
    valid_peaks = peaks[~is_buildup]
    valid_h = peak_h[~is_buildup]
    
    peak_labels = []
    if len(valid_peaks) == 0:
        return ["Build-up"] * len(peaks), "Build-up", peaks, ["Build-up"] * len(peaks)
        
    sorted_idx = np.argsort(valid_h)[::-1]
    clusters = []
    current_cluster = [sorted_idx[0]]
    
    for idx in sorted_idx[1:]:
        curr_mean = valid_h[current_cluster].mean()
        if abs(valid_h[idx] - curr_mean) < 0.05 * amp:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)
    
    tier_names = ["Tall", "Short", "Mini"] if len(clusters) <= 3 else [f"Tier {i+1}" for i in range(len(clusters))]
    if len(clusters) == 3: tier_names = ["Tall", "Medium", "Short"]
    
    cluster_labels = {}
    for i, cluster in enumerate(clusters):
        label = tier_names[i] if i < len(tier_names) else f"Tier {i+1}"
        for idx in cluster:
            cluster_labels[idx] = label
            
    valid_labels = [cluster_labels[i] for i in range(len(valid_peaks))]
    
    pattern_labels = valid_labels
    pattern_str = "-".join(pattern_labels)
    for i in range(1, len(pattern_labels)//2 + 1):
        if len(pattern_labels) % i == 0:
            repeats = len(pattern_labels) // i
            unit = pattern_labels[:i]
            if pattern_labels == unit * repeats:
                pattern_str = "-".join(unit)
                break
                
    valid_idx = 0
    for i in range(len(peaks)):
        if is_buildup[i]:
            peak_labels.append("Build-up")
        else:
            peak_labels.append(valid_labels[valid_idx])
            valid_idx += 1
            
    return peak_labels, pattern_str, valid_peaks, valid_labels

def _compute_face_angles(theta_deg, h, peaks, peak_labels, pattern_str, valid_peaks, valid_labels, r_shell):
    """Compute exact Cartesian face angles relative to the tip radial vector."""
    angles_by_label = {}
    n_primary = len(valid_peaks)
    
    if n_primary > 0:
        for i, pk in enumerate(valid_peaks):
            label = valid_labels[i]
            if label not in angles_by_label:
                angles_by_label[label] = {"lead": [], "trail": []}
                
            pk_prev = valid_peaks[(i - 1) % n_primary]
            pk_next = valid_peaks[(i + 1) % n_primary]
            
            v_left = get_true_valley(h, pk, pk_prev, -1)
            v_right = get_true_valley(h, pk, pk_next, 1)
            
            def _svd_angle(th_seg, h_seg, r_shell):
                """Compute face angle using SVD. Falls back to two-point vector for near-vertical faces."""
                dh = h_seg.max() - h_seg.min()
                if dh <= 0.15 * (h.max() - h.min()) or len(th_seg) < 4:
                    return float('nan')
                # Primary: use middle 20-80% amplitude band for a robust SVD fit
                mask = (h_seg >= h_seg.min() + 0.2*dh) & (h_seg <= h_seg.min() + 0.8*dh)
                if mask.sum() >= 3:
                    pts = mask
                else:
                    # Fallback for near-vertical / step faces:
                    # Find the exact drop/rise by looking at 80% and 20% height crossings
                    h_80 = h_seg.min() + 0.8 * dh
                    h_20 = h_seg.min() + 0.2 * dh
                    tip_i_abs = np.argmax(h_seg)
                    bot_i_abs = np.argmin(h_seg)
                    
                    if tip_i_abs < bot_i_abs:
                        # Falling face (Trailing)
                        try:
                            top_i = np.where(h_seg >= h_80)[0][-1]
                            bot_i = np.where(h_seg <= h_20)[0][0]
                            pts_idx = np.array([top_i, bot_i])
                        except IndexError:
                            pts_idx = np.array([tip_i_abs, bot_i_abs])
                    else:
                        # Rising face (Leading)
                        try:
                            bot_i = np.where(h_seg <= h_20)[0][-1]
                            top_i = np.where(h_seg >= h_80)[0][0]
                            pts_idx = np.array([top_i, bot_i])
                        except IndexError:
                            pts_idx = np.array([tip_i_abs, bot_i_abs])

                    # Build a two-point direction vector in Cartesian space
                    r_pts = r_shell - h_seg[pts_idx]
                    th_pts = np.radians(th_seg[pts_idx])
                    x_pts = r_pts * np.cos(th_pts)
                    y_pts = r_pts * np.sin(th_pts)
                    v_face = np.array([x_pts[0] - x_pts[1], y_pts[0] - y_pts[1]], dtype=float)
                    norm = np.linalg.norm(v_face)
                    if norm < 1e-9:
                        return float('nan')
                    v_face /= norm
                    tip_idx = np.argmax(h_seg)
                    r_tip = np.array([
                        (r_shell - h_seg[tip_idx]) * np.cos(np.radians(th_seg[tip_idx])),
                        (r_shell - h_seg[tip_idx]) * np.sin(np.radians(th_seg[tip_idx])),
                    ])
                    r_tip /= np.linalg.norm(r_tip)
                    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v_face, r_tip)), 0, 1))))
                # Standard SVD path
                r_face = r_shell - h_seg[pts]
                th_rad = np.radians(th_seg[pts])
                x_val, y_val = r_face * np.cos(th_rad), r_face * np.sin(th_rad)
                coords = np.vstack([x_val, y_val]).T
                _, _, Vt = np.linalg.svd(coords - coords.mean(axis=0))
                v_face = Vt[0]
                tip_idx = np.argmax(h_seg)
                r_tip = np.array([
                    (r_shell - h_seg[tip_idx]) * np.cos(np.radians(th_seg[tip_idx])),
                    (r_shell - h_seg[tip_idx]) * np.sin(np.radians(th_seg[tip_idx])),
                ])
                r_tip /= np.linalg.norm(r_tip)
                return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v_face, r_tip)), 0, 1))))

            # Leading face
            if v_left <= pk: th_lead, h_lead = theta_deg[v_left:pk+1], h[v_left:pk+1]
            else: th_lead, h_lead = np.concatenate([theta_deg[v_left:], theta_deg[:pk+1]+360]), np.concatenate([h[v_left:], h[:pk+1]])
            ang = _svd_angle(th_lead, h_lead, r_shell)
            if not np.isnan(ang):
                angles_by_label[label]["lead"].append(ang)

            # Trailing face
            if pk <= v_right: th_trail, h_trail = theta_deg[pk:v_right+1], h[pk:v_right+1]
            else: th_trail, h_trail = np.concatenate([theta_deg[pk:], theta_deg[:v_right+1]+360]), np.concatenate([h[pk:], h[:v_right+1]])
            ang = _svd_angle(th_trail, h_trail, r_shell)
            if not np.isnan(ang):
                angles_by_label[label]["trail"].append(ang)

    result = {}
    for label, angs in angles_by_label.items():
        result[f"{label} Leading Angle"] = float(np.median(angs["lead"])) if angs["lead"] else float('nan')
        result[f"{label} Trailing Angle"] = float(np.median(angs["trail"])) if angs["trail"] else float('nan')
        
    # Generate the Face Detection Plot
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(theta_deg, h, 'k-', linewidth=1.2, label='Liner Profile')
    
    added_lead = False
    added_trail = False
    
    if n_primary > 0:
        for i, pk in enumerate(valid_peaks):
            label = valid_labels[i]
            pk_prev = valid_peaks[(i - 1) % n_primary]
            pk_next = valid_peaks[(i + 1) % n_primary]
            v_left = get_true_valley(h, pk, pk_prev, -1)
            v_right = get_true_valley(h, pk, pk_next, 1)
            
            # Highlight Lead
            if v_left <= pk: th_lead, h_lead = theta_deg[v_left:pk+1], h[v_left:pk+1]
            else: th_lead, h_lead = np.concatenate([theta_deg[v_left:], theta_deg[:pk+1]+360]), np.concatenate([h[v_left:], h[:pk+1]])
            label_lead = 'Leading Face' if not added_lead else ""
            ax.fill_between(th_lead, h_lead, h.min(), color='#ef4444', alpha=0.5, label=label_lead)
            added_lead = True
            
            # Highlight Trail
            if pk <= v_right: th_trail, h_trail = theta_deg[pk:v_right+1], h[pk:v_right+1]
            else: th_trail, h_trail = np.concatenate([theta_deg[pk:], theta_deg[:v_right+1]+360]), np.concatenate([h[pk:], h[:v_right+1]])
            label_trail = 'Trailing Face' if not added_trail else ""
            ax.fill_between(th_trail, h_trail, h.min(), color='#3b82f6', alpha=0.5, label=label_trail)
            added_trail = True
            
    # Mark peaks with labels
    for pk, lbl in zip(peaks, peak_labels):
        color = 'black' if lbl == 'Build-up' else 'green'
        marker = 'x' if lbl == 'Build-up' else 'v'
        ax.plot(theta_deg[pk], h[pk] + 5, marker=marker, color=color, markersize=8)
        ax.text(theta_deg[pk], h[pk] + 15, f"[{lbl}]", ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
        
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Protrusion (mm)')
    ax.set_title(f'Face Angle Detection Regions | Pattern Detected: {pattern_str}')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if n_primary > 10:
        ax.set_xlim(0, max(90, theta_deg[valid_peaks[min(3, len(valid_peaks)-1)]] + 10))
    else:
        ax.set_xlim(0, 360)
        
    ax.set_ylim(bottom=h.min() - 10, top=h.max() + 50)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    
    return result, buf.getvalue()

def extract_ml_features_from_angular_df(angular_df, n_harmonics=50):
    theta_deg   = angular_df["angle_deg"].values
    h           = angular_df["protrusion_mm"].values
    h_ext = np.concatenate([h, h, h])
    amp   = h.max() - h.min()
    pks_ext, _ = find_peaks(h_ext, prominence=0.2 * amp, distance=3)
    n = len(h)
    n_total_lifters = int(len(
        np.unique(pks_ext[(pks_ext >= n) & (pks_ext < 2 * n)] - n)
    ))
    period_deg, _   = _true_period_via_autocorr(h)
    n_repeat_units  = int(round(360.0 / period_deg))
    n_lifters_pu    = n_total_lifters / n_repeat_units if n_repeat_units else float("nan")
    
    # Nyquist limit scaling: ensure resolution is at least 2 * n_harmonics + buffer
    n_samples = max(128, n_harmonics * 2 + 10)
    
    h_unit    = _extract_unit_shape_vector(theta_deg, h, period_deg, n_samples)
    shape_mag = _fourier_magnitude(h_unit, n_harmonics)
    result = {
        "n_total_lifters":    n_total_lifters,
        "n_repeat_units":     n_repeat_units,
        "n_lifters_per_unit": n_lifters_pu,
    }
    r_shell = angular_df["r_inner_mm"].iloc[0] + angular_df["protrusion_mm"].iloc[0]
    peak_labels, pattern_str, valid_peaks, valid_labels = cluster_lifter_patterns(h, pks_ext[(pks_ext >= n) & (pks_ext < 2 * n)] - n)
    result["lifter_pattern"] = pattern_str
    
    angles_dict, face_plot_bytes = _compute_face_angles(theta_deg, h, np.unique(pks_ext[(pks_ext >= n) & (pks_ext < 2 * n)] - n), peak_labels, pattern_str, valid_peaks, valid_labels, r_shell)
    
    for k, v in angles_dict.items():
        result[k] = v
        
    # Keep standard keys for UI logging compatibility
    result["leading_face_angle"]       = angles_dict.get("Tall Leading Angle",  float('nan'))
    result["trailing_face_angle"]      = angles_dict.get("Tall Trailing Angle", float('nan'))
    result["short_leading_face_angle"] = angles_dict.get("Short Leading Angle", float('nan'))
    result["short_trailing_face_angle"]= angles_dict.get("Short Trailing Angle",float('nan'))
    result["_face_plot_bytes"] = face_plot_bytes
    
    for k, v in enumerate(shape_mag):
        result[f"shape_k{k}"] = float(v)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DEM CSV processing
# ─────────────────────────────────────────────────────────────────────────────
def parse_dem_csv_bytes(content_bytes: bytes):
    text = content_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.reader(text.splitlines())
    headers, rows = [], []
    found_header = False
    for raw_row in reader:
        if not any(raw_row):
            continue
            
        if not found_header:
            # Robust header detection: look for key column names
            lower_row = [str(x).lower() for x in raw_row]
            has_time = any('time' in h for h in lower_row)
            has_cf   = any('compressive' in h for h in lower_row)
            has_pow  = any('power' in h for h in lower_row)
            
            if has_time and (has_cf or has_pow):
                headers = [h.strip() for h in raw_row]
                found_header = True
            continue
            
        try:
            # Only parse if we have a matching number of columns
            if len(raw_row) >= len(headers) and any(raw_row):
                # We slice raw_row to match len(headers) in case there are trailing empty commas
                sliced_row = raw_row[:len(headers)]
                rows.append({h: float(v.strip()) for h, v in zip(headers, sliced_row) if h})
        except ValueError:
            # Skip rows that cannot be converted to floats (e.g., secondary headers, text notes)
            continue
    return headers, rows


def convert_dem_to_rotation_pct(content_bytes: bytes, target_rpm: float):
    if target_rpm <= 0:
        raise ValueError("Invalid Mill RPM. Please enter a positive RPM.")
    headers, rows = parse_dem_csv_bytes(content_bytes)
    if not rows:
        raise ValueError("Failed to parse DEM CSV: No numeric data rows found. Check if the file contains the correct headers and numeric values.")
    def _find_col(kw):
        m = [h for h in headers if kw.lower() in h.lower()]
        if not m:
            raise KeyError(f"No column containing '{kw}'. Headers: {headers}")
        return m[0]

    time_key  = _find_col('time')

    cf_key    = _find_col('Compressive Force')
    ke_key    = _find_col('Kinetic Energy')
    power_key = _find_col('Geometry Power')
    times   = [r[time_key] for r in rows]
    t_start, t_end = times[0], times[-1]
    t_span  = t_end - t_start
    if t_span <= 0:
        raise ValueError(f"Invalid time span in CSV: {t_span:.6f} s")
        
    t_full_rot = 60.0 / target_rpm
    
    # Always set the target window as the LAST full rotation equivalent (closest to steady-state)
    t_sample_start = t_end - t_full_rot
    t1  = t_full_rot / 100.0
    
    out = []
    for p in range(1, 101):
        t_target = t_sample_start + p * t1
        
        # Modulo Time-Wrapper: Wrap the target time back into the available window if it falls outside
        if t_target < t_start:
            offset = (t_target - t_start) % t_span
            t_target_wrapped = t_start + offset
        else:
            t_target_wrapped = t_target
            
        chosen   = min(rows, key=lambda r: abs(r[time_key] - t_target_wrapped))
        out.append({
            "pct_rotation":            p,
            "cf_max_particle":         chosen[cf_key],
            "ke_max_particle":         chosen[ke_key],
            "power_total_geometry_kw": round(abs(chosen[power_key]) / 1000, 6),
        })
    return pd.DataFrame(out), target_rpm, (t_span / t_full_rot)


# ─────────────────────────────────────────────────────────────────────────────
# Plot generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_liner_profile_png(all_points, theta_centers, r_inner, r_outer,
                                h_protrusion, peaks, r_shell, center_2d):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#ffffff")
    fig.suptitle("Liner Geometry Profile", color="#1c2128", fontsize=13, fontweight='bold')
    for ax in (ax1, ax2):
        ax.set_facecolor("#f8fafc")
        for sp in ax.spines.values():
            sp.set_color("#d0d7de")
        ax.tick_params(colors="#57606a")
        ax.xaxis.label.set_color("#57606a")
        ax.yaxis.label.set_color("#57606a")
        ax.title.set_color("#1c2128")
    ax1.scatter(all_points[:, 0], all_points[:, 1], s=0.1, c='#2563eb', alpha=0.12)
    x_inner = r_inner * np.cos(theta_centers) + center_2d[0]
    y_inner = r_inner * np.sin(theta_centers) + center_2d[1]
    ax1.plot(x_inner, y_inner, '#2563eb', linewidth=0.8, label='Inner profile')
    for pk in peaks:
        ax1.plot(x_inner[pk], y_inner[pk], 'v', color='#16a34a', markersize=5)
    theta_c = np.linspace(0, 2 * np.pi, 500)
    ax1.plot(r_shell * np.cos(theta_c) + center_2d[0],
             r_shell * np.sin(theta_c) + center_2d[1],
             color='#9ca3af', linewidth=0.6, linestyle='--')
    ax1.set_aspect('equal')
    ax1.set_title(f"Cross-Section  ({len(peaks)} lifters detected)")
    ax1.legend(fontsize=7, framealpha=0.7)
    theta_deg = np.degrees(theta_centers)
    ax2.fill_between(theta_deg, 0, h_protrusion, color='#2563eb', alpha=0.12)
    ax2.plot(theta_deg, h_protrusion, '#2563eb', linewidth=0.9)
    for pk in peaks:
        ax2.plot(theta_deg[pk], h_protrusion[pk], 'v', color='#16a34a', markersize=6)
    ax2.axhline(np.mean(h_protrusion), color='#d97706', linewidth=1.0,
                linestyle='--', label=f"Mean {np.mean(h_protrusion):.1f} mm")
    ax2.set_xlabel("Angle (degrees)")
    ax2.set_ylabel("Protrusion height (mm)")
    ax2.set_title("Protrusion h(θ)")
    ax2.set_xlim(0, 360)
    ax2.legend(fontsize=7, framealpha=0.7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches='tight', facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_full_analysis_png(all_points, theta_centers, r_inner, r_outer,
                                h_protrusion, thickness, peaks, r_shell,
                                center_2d, s_uniform, h_arc, mag, n_lifters,
                                dominant_k, n_solids, mill_axis, plane_name,
                                section_offset, lifter_spacings_deg):
    fig = plt.figure(figsize=(24, 20), facecolor="#ffffff")
    fig.suptitle(
        f"{n_lifters} Lifters Detected  |  Mill axis: {mill_axis}  |  "
        f"Section: {plane_name} @ {section_offset:.1f} mm",
        color="#1c2128", fontsize=14, fontweight='bold', y=0.99)

    def _style_ax(ax, polar=False):
        ax.set_facecolor("#f8fafc")
        if not polar:
            for sp in ax.spines.values():
                sp.set_color("#d0d7de")
        ax.tick_params(colors="#57606a", labelsize=7)
        ax.xaxis.label.set_color("#57606a") if not polar else None
        ax.yaxis.label.set_color("#57606a") if not polar else None
        ax.title.set_color("#1c2128")

    theta_deg = np.degrees(theta_centers)
    n_harmonics = min(120, len(mag))

    ax1 = fig.add_subplot(3, 3, 1, aspect='equal')
    _style_ax(ax1)
    ax1.scatter(all_points[:, 0], all_points[:, 1], s=0.05, c='#2563eb', alpha=0.08)
    x_in = r_inner * np.cos(theta_centers) + center_2d[0]
    y_in = r_inner * np.sin(theta_centers) + center_2d[1]
    ax1.plot(x_in, y_in, '#2563eb', linewidth=0.5, label='Inner')
    for pk in peaks:
        ax1.plot(x_in[pk], y_in[pk], 'v', color='#16a34a', markersize=4)
    ax1.set_title(f"Cross-Section ({n_lifters} lifters)")
    ax1.legend(fontsize=5)

    ax2 = fig.add_subplot(3, 3, 2, projection='polar')
    _style_ax(ax2, polar=True)
    ax2.plot(theta_centers, r_inner, '#2563eb', linewidth=0.5)
    ax2.fill_between(theta_centers, r_inner, r_shell, alpha=0.08, color='#2563eb')
    for pk in peaks:
        ax2.plot(theta_centers[pk], r_inner[pk], 'v', color='#dc2626', markersize=3)
    ax2.set_title("Inner Profile (Polar)\n", pad=18, color='#1c2128', fontsize=9)

    ax3 = fig.add_subplot(3, 3, 3)
    _style_ax(ax3)
    ax3.plot(theta_deg, h_protrusion, '#2563eb', linewidth=0.5)
    for pk in peaks:
        ax3.plot(theta_deg[pk], h_protrusion[pk], 'v', color='#dc2626', markersize=3)
    ax3.set_xlabel("Angle (degrees)")
    ax3.set_ylabel("Protrusion (mm)")
    ax3.set_title(f"h(θ)  –  {n_lifters} lifters")
    ax3.set_xlim(0, 360)

    ax4 = fig.add_subplot(3, 3, 4)
    _style_ax(ax4)
    m90 = theta_deg < 90
    ax4.plot(theta_deg[m90], h_protrusion[m90], '#2563eb', linewidth=0.6)
    for pk in peaks[theta_deg[peaks] < 90]:
        ax4.plot(theta_deg[pk], h_protrusion[pk], 'v', color='#dc2626', markersize=5)
    ax4.set_xlabel("Angle (degrees)")
    ax4.set_ylabel("Protrusion (mm)")
    ax4.set_title("Zoomed 0–90°")
    ax4.set_xlim(0, 90)

    ax5 = fig.add_subplot(3, 3, 5)
    _style_ax(ax5)
    
    # Calculate segment width based on actual detected lifters, not CAD solids
    num_peaks = len(peaks)
    # Use a 2.5x multiplier to ensure at least 2 full lifters are shown (crucial for High-Low lifter designs)
    seg = min((360.0 / num_peaks) * 2.1 if num_peaks > 0 else 60.0, 180.0)
    
    ms  = theta_deg < seg
    ax5.plot(theta_deg[ms], h_protrusion[ms], '#2563eb', linewidth=0.8, marker='.', markersize=1)
    for pk in peaks[theta_deg[peaks] < seg]:
        ax5.plot(theta_deg[pk], h_protrusion[pk], 'v', color='#dc2626', markersize=7)
    ax5.set_xlabel("Angle (degrees)")
    ax5.set_ylabel("Protrusion (mm)")
    ax5.set_title("Zoomed ~1 segment")
    ax5.set_xlim(0, seg)

    ax6 = fig.add_subplot(3, 3, 6)
    _style_ax(ax6)
    ax6.plot(s_uniform, h_arc, '#2563eb', linewidth=0.6)
    ax6.axhline(np.mean(h_arc), color='#d97706', linestyle='--', linewidth=0.9,
                label=f"Mean {np.mean(h_arc):.1f} mm")
    ax6.set_xlabel("Normalized arc length")
    ax6.set_ylabel("Protrusion (mm)")
    ax6.set_title("h(s) – Arc-Length Profile")
    ax6.legend(fontsize=6)

    ax7 = fig.add_subplot(3, 3, 7)
    _style_ax(ax7)
    k_vals = np.arange(1, n_harmonics)
    colors = ['#16a34a' if k == n_lifters else '#dc2626' if k == dominant_k else '#2563eb' for k in k_vals]
    ax7.bar(k_vals, mag[1:n_harmonics], color=colors, edgecolor='none', linewidth=0)
    if 0 < n_lifters < n_harmonics:
        ax7.axvline(n_lifters, color='#16a34a', linestyle='--', linewidth=1.2,
                    label=f'k={n_lifters} (lifters)')
    ax7.set_xlabel("Harmonic k")
    ax7.set_ylabel("|Coeff| (mm)")
    ax7.set_title(f"Fourier Spectrum  (dominant k={dominant_k})")
    ax7.legend(fontsize=6)

    ax8 = fig.add_subplot(3, 3, 8)
    _style_ax(ax8)
    if len(lifter_spacings_deg) > 1:
        ax8.hist(lifter_spacings_deg, bins=max(5, min(30, n_lifters)),
                 color='#2563eb', edgecolor='#ffffff', linewidth=0.4)
        if n_lifters > 0:
            ax8.axvline(360 / n_lifters, color='#dc2626', linestyle='--', linewidth=1.2,
                        label=f"Expected {360/n_lifters:.1f}°")
        ax8.set_xlabel("Spacing (degrees)")
        ax8.set_ylabel("Count")
        ax8.set_title("Lifter Spacing Distribution")
        ax8.legend(fontsize=6)
    else:
        ax8.text(0.5, 0.5, "Not enough lifters\nfor spacing analysis",
                 ha='center', va='center', transform=ax8.transAxes,
                 color='#57606a', fontsize=11)
        ax8.set_title("Lifter Spacing Distribution")

    ax9 = fig.add_subplot(3, 3, 9)
    _style_ax(ax9)
    ax9.plot(theta_deg, thickness, '#16a34a', linewidth=0.5)
    ax9.axhline(np.median(thickness), color='#d97706', linestyle='--', linewidth=0.9,
                label=f"Median {np.median(thickness):.0f} mm")
    ax9.set_xlabel("Angle (degrees)")
    ax9.set_ylabel("Thickness (mm)")
    ax9.set_title("Liner Thickness")
    ax9.set_xlim(0, 360)
    ax9.legend(fontsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches='tight', facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Full STEP analysis pipeline (with verbose log)
# ─────────────────────────────────────────────────────────────────────────────
# Module-level handle to the running analysis thread so Stop can target it.



def analyse_step_process_target(step_path: str, q: mp.Queue):
    def L(level, msg):
        q.put({"type": "log", "level": level, "msg": msg})
        if level == "inf":
            txt = msg.lower()
            if "loading step" in txt: p = 5
            elif "detecting mill axis" in txt: p = 15
            elif "sectioning" in txt: p = 25
            elif "polar profile" in txt: p = 40
            elif "lifter peaks" in txt: p = 55
            elif "fourier" in txt: p = 70
            elif "dataframe" in txt: p = 85
            elif "extracting ml" in txt: p = 90
            elif "rendering" in txt: p = 95
            else: p = -1
            if p != -1:
                q.put({"type": "progress", "val": p, "stage": msg})
        elif level == "ok" and "complete" in msg.lower():
            q.put({"type": "progress", "val": 100})
        time.sleep(0.05)

    try:
        L("inf", "Initializing geometry engine (may take 15s on first run)...")
        try:
            import cadquery as cq
        except Exception as e:
            err_msg = f"CadQuery / OCP geometry engine could not be initialized: {e}"
            L("err", err_msg)
            q.put({"type": "error", "error": err_msg})
            return False
        
        L("inf", f"Loading STEP file: {os.path.basename(step_path)}")
        result = cq.importers.importStep(step_path)
        solids = result.solids().vals()
        if not solids:
            err_msg = "No solids found in STEP file."
            L("err", err_msg)
            q.put({"type": "error", "error": err_msg})
            return False
        L("ok", f"Loaded {len(solids)} solid(s) from file.")

        L("inf", "Detecting mill axis (shortest bounding-box span)...")
        mill_axis, plane_name, plane_axes, center_3d, bbox_min, bbox_max = _detect_mill_axis(solids)
        axis_names = ['X', 'Y', 'Z']
        mill_idx   = axis_names.index(mill_axis)
        section_offset = center_3d[mill_idx]
        spans = bbox_max - bbox_min
        L("ok", f"Mill axis: {mill_axis}  |  Cross-section plane: {plane_name}  |  Offset: {section_offset:.2f} mm")
        L("inf", f"   Bounding box spans - X:{spans[0]:.1f}  Y:{spans[1]:.1f}  Z:{spans[2]:.1f} mm")

        L("inf", f" Sectioning {len(solids)} solid(s) at midplane...")
        def section_log(level, msg): L(level, msg)
        all_points = _collect_section_points(solids, plane_name, section_offset, plane_axes, section_log)
        if len(all_points) == 0:
            L("err", "No section points collected - the midplane may miss geometry.")
            return False
        L("ok", f"Collected {len(all_points):,} cross-section points.")

        L("inf", "Building polar profile (32768 angular bins for extreme precision)...")
        (theta_centers, r_inner, r_outer, h_protrusion, thickness,
         r_shell, center_2d) = _build_polar_profile(all_points)
        L("ok", f"Shell radius: {r_shell:.2f} mm  |  Max protrusion: {h_protrusion.max():.2f} mm  |  Min: {h_protrusion.min():.2f} mm")

        L("inf", "Detecting lifter peaks (multi-parameter sweep)...")
        peaks    = _detect_lifters(h_protrusion, len(theta_centers))
        n_lifters = len(peaks)
        lifter_angles_deg    = np.degrees(theta_centers[peaks])
        lifter_spacings_deg  = np.diff(lifter_angles_deg) if n_lifters > 1 else np.array([])
        L("ok", f"{n_lifters} lifters detected.")
        if n_lifters > 1:
            L("inf", f"   Average spacing: {lifter_spacings_deg.mean():.2f}° ± {lifter_spacings_deg.std():.2f}°  (expected: {360/n_lifters:.2f}°)")

        L("inf", "Running arc-length sampling and Fourier analysis...")
        s_uniform, h_arc, _ = _arc_length_sample(theta_centers, r_inner, h_protrusion)
        mag, dominant_k      = _fourier_analysis(h_arc)
        L("ok", f"Fourier done - dominant harmonic k={dominant_k}.")

        L("inf", "Building angular profile DataFrame...")
        ang_df = pd.DataFrame({
            "angle_deg":     np.degrees(theta_centers),
            "r_inner_mm":    r_inner,
            "r_outer_mm":    r_outer,
            "thickness_mm":  thickness,
            "protrusion_mm": h_protrusion,
        })
        L("ok", f"Angular profile: {len(ang_df)} rows.")

        L("inf", "Extracting ML geometry features...")
        ml_feats = extract_ml_features_from_angular_df(ang_df, int(80))

        log_str = (f"ML features: {ml_feats['n_total_lifters']} lifters | {ml_feats['n_repeat_units']} repeat units | "
                   f"{ml_feats['n_lifters_per_unit']:.2f} lifters/unit | Pattern: {ml_feats.get('lifter_pattern', 'Unknown')}")
        L("ok", log_str)
        for k, v in ml_feats.items():
            if "Angle" in k and not pd.isna(v):
                L("inf", f"   {k}: {v:.1f}°")

        L("inf", " Rendering plots...")
        liner_png = generate_liner_profile_png(
            all_points, theta_centers, r_inner, r_outer,
            h_protrusion, peaks, r_shell, center_2d)
        full_png = generate_full_analysis_png(
            all_points, theta_centers, r_inner, r_outer,
            h_protrusion, thickness, peaks, r_shell, center_2d,
            s_uniform, h_arc, mag, n_lifters, dominant_k,
            len(solids), mill_axis, plane_name, section_offset,
            lifter_spacings_deg)
        L("ok", "Plots rendered successfully.")

        face_img_bytes = ml_feats.pop("_face_plot_bytes", None)
        d_theta = theta_centers[1] - theta_centers[0]
        total_area = np.pi * (r_shell ** 2)
        liner_area = np.sum(0.5 * (r_outer**2 - r_inner**2) * d_theta)
        eff_area = total_area - liner_area
        r_eff = np.sqrt(eff_area / np.pi) if eff_area > 0 else r_shell
        suggested_dia_m = round((2.0 * r_eff) / 1000.0, 4)

        L("inf", "--- Cross-Sectional Area Analysis ---")
        L("inf", f"   Outer Diameter of Shell : {(2.0 * r_shell):.2f} mm")
        L("inf", f"   Total Cross-Sectional Area : {total_area:.2f} mm²")
        L("inf", f"   Liner Profile Area : {liner_area:.2f} mm²")
        L("inf", f"   Effective Mill Area (Total - Liner) : {eff_area:.2f} mm²")
        L("inf", f"   Calculated Effective Diameter : {(2.0 * r_eff):.2f} mm ({suggested_dia_m:.4f} m)")
        L("inf", "---------------------------------------")

        res = {
            "angular_profile_df": ang_df,
            "face_analysis_img": face_img_bytes,
            "ml_features": ml_feats,
            "liner_profile_img": liner_png,
            "analysis_img": full_png,
            "n_lifters_detected": n_lifters,
            "step_file_path": step_path,
            "suggested_mill_dia": suggested_dia_m
        }
        L("ok", f"Analysis complete - {n_lifters} lifters detected.")
        q.put({"type": "done", "results": res})

    except Exception as e:
        q.put({"type": "error", "error": f"Analysis failed: {str(e)}\n{traceback.format_exc()}"})


def start_analysis_process(step_path):
    import streamlit as st
    st.session_state["is_analysing"] = True
    st.session_state["analysis_just_finished"] = False
    st.session_state["cancel_analysis"] = False
    st.session_state["analysis_log"] = []
    st.session_state["analysis_progress"] = 0
    st.session_state["analysis_stage"] = "Starting..."

    q = mp.Queue()
    st.session_state["_analysis_queue"] = q
    p = mp.Process(target=analyse_step_process_target, args=(step_path, q), daemon=True)
    st.session_state["_analysis_process"] = p
    p.start()
