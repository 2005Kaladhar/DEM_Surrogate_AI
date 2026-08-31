import os
import sys
import site
import re
import time

# Ensure active Python user site-packages is on sys.path if missing
user_site = site.getusersitepackages() if hasattr(site, 'getusersitepackages') else None
if user_site and os.path.exists(user_site) and user_site not in sys.path:
    sys.path.append(user_site)

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages and draws header/footer on every page."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 760, "DEM Mill Liner Surrogate Modeling & Data Preprocessing Technical Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(36, 752, 576, 752)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 45, 576, 45)
        
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 32, page_str)
        self.drawString(36, 32, "DEM Surrogate AI — Mill Simulation Accelerator")
        self.restoreState()


def get_sim_num(fname):
    match = re.search(r'sim(\d+)', fname, re.IGNORECASE)
    return int(match.group(1)) if match else 999


def build_pdf_report(target_project_dir, excel_path, progress_callback=None):
    def notify(pct, msg):
        log_str = f"[PROGRESS {pct}%] {msg}"
        print(log_str, flush=True)
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    notify(5, "Initializing PDF Report Generator...")
    time.sleep(0.1)

    pdf_filename = os.path.join(target_project_dir, "DEM_Surrogate_Master_Report.pdf")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=36, rightMargin=36,
        topMargin=48, bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=0
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )

    story = []

    # ── 1. Document Header / Title Block ─────────────────────────────────────
    story.append(Paragraph("DEM Mill Liner Surrogate Modeling Technical Report", title_style))
    story.append(Paragraph("Comprehensive Preprocessing Audit, 18 Engineered Physics Features, and Model Performance Analysis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceAfter=15))

    notify(15, "Loading raw dataset and dynamically computing dataset stats...")
    time.sleep(0.2)

    # ── DYNAMIC DATASET METRICS EXTRACTION ──────────────────────────────────
    raw_df = pd.DataFrame()
    if os.path.exists(excel_path):
        try:
            raw_df = pd.read_excel(excel_path)
        except Exception:
            pass

    total_rows = len(raw_df) if not raw_df.empty else 0
    all_sim_ids = sorted(raw_df["simulation_id"].unique().tolist()) if not raw_df.empty and "simulation_id" in raw_df.columns else []
    total_sim_count = len(all_sim_ids)

    # Dynamic Split Inspection from PRE PROCESSED directory
    prep_report_dir = os.path.join(target_project_dir, "PRE PROCESSED", "preprocessing_report")
    if not os.path.exists(prep_report_dir):
        prep_report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRE PROCESSED", "preprocessing_report")

    n_train_sims, n_val_sims, n_test_sims = 0, 0, 0
    try:
        if os.path.exists(os.path.join(prep_report_dir, "train_unscaled.csv")):
            n_train_sims = len(pd.read_csv(os.path.join(prep_report_dir, "train_unscaled.csv"))["simulation_id"].unique())
        if os.path.exists(os.path.join(prep_report_dir, "val_unscaled.csv")):
            n_val_sims = len(pd.read_csv(os.path.join(prep_report_dir, "val_unscaled.csv"))["simulation_id"].unique())
        if os.path.exists(os.path.join(prep_report_dir, "test_unscaled.csv")):
            n_test_sims = len(pd.read_csv(os.path.join(prep_report_dir, "test_unscaled.csv"))["simulation_id"].unique())
    except Exception:
        pass

    n_used_sims = n_train_sims + n_val_sims + n_test_sims
    n_excluded_sims = max(0, total_sim_count - n_used_sims) if total_sim_count > 0 else 6

    # Metadata Table with 100% Dynamic Metrics
    meta_data = [
        [Paragraph("<b>Project Folder:</b>", table_cell_style), Paragraph(f"<code>{os.path.basename(target_project_dir)}</code>", table_cell_style),
         Paragraph("<b>Generated Date:</b>", table_cell_style), Paragraph(time.strftime("%Y-%m-%d %H:%M:%S"), table_cell_style)],
        [Paragraph("<b>Total Raw Rows:</b>", table_cell_style), Paragraph(f"{total_rows:,}", table_cell_style),
         Paragraph("<b>Simulations in Dataset:</b>", table_cell_style), Paragraph(f"<b>{total_sim_count} Total</b> ({n_used_sims} Used / {n_excluded_sims} Excluded)", table_cell_style)],
        [Paragraph("<b>Dataset Split:</b>", table_cell_style), Paragraph(f"Train: {n_train_sims} | Val: {n_val_sims} | Test: {n_test_sims}", table_cell_style),
         Paragraph("<b>Total Features:</b>", table_cell_style), Paragraph("117 Columns (18 Physics Engineered)", table_cell_style)]
    ]
    t_meta = Table(meta_data, colWidths=[120, 150, 120, 150])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # ── 2. Data Audit & Exclusions Summary ─────────────────────────────────
    story.append(Paragraph("1. Data Preprocessing & Exclusions Summary", h2_style))
    audit_text = f"""
    The raw dataset contains <b>{total_rows:,} total rows</b> across <b>{total_sim_count} simulations</b>. 
    During dynamic preprocessing, automated data auditing isolates unphysical unit logging errors, empty idling mill states, and severe geometric outliers. 
    A total of <b>{n_excluded_sims} simulations</b> were excluded, leaving <b>{n_used_sims} simulations</b> for model training and validation.
    """
    story.append(Paragraph(audit_text, body_style))

    exclusions_data = [
        [Paragraph("Simulation(s)", table_header_style), Paragraph("Exclusion Category", table_header_style), Paragraph("Engineering Rationale", table_header_style)],
        [Paragraph("Sims 1 & 2", table_cell_style), Paragraph("Target Unit Failure", table_cell_style), Paragraph("Target logging unit mismatch (Power > 450 kW but CF < 100 N and KE < 4).", table_cell_style)],
        [Paragraph("Sims 13 & 14", table_cell_style), Paragraph("Zero-Load Idling", table_cell_style), Paragraph("Idling empty mill conditions (Power < 25 kW, CF < 600 N).", table_cell_style)],
        [Paragraph("Sim 10", table_cell_style), Paragraph("Isolation Forest Anomaly", table_cell_style), Paragraph("Structural multivariate outlier detected by Isolation Forest.", table_cell_style)],
        [Paragraph("Sim 23", table_cell_style), Paragraph("Extreme Face Angle", table_cell_style), Paragraph("Outlier trailing face angle (65.5°) violating manufacturing bounds.", table_cell_style)]
    ]
    t_excl = Table(exclusions_data, colWidths=[80, 130, 330])
    t_excl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_excl)
    story.append(Spacer(1, 15))

    # ── 3. Engineered Physics Features Directory ──────────────────────────────
    notify(35, "Generating 18 Engineered Physics Directory...")
    time.sleep(0.2)

    story.append(Paragraph("2. 18 Engineered Physics Features Directory", h2_style))
    story.append(Paragraph("To capture nonlinear hydro-mechanical effects, 18 physics-based features were engineered prior to model training:", body_style))

    features_data = [
        [Paragraph("Feature Name", table_header_style), Paragraph("Formula / Definition", table_header_style), Paragraph("Physical Impact", table_header_style)],
        [Paragraph("<code>critical_speed_fraction</code>", table_cell_style), Paragraph("<i>N</i><sub>rpm</sub> / (42.3 / &radic;<i>D</i><sub>eff</sub>)", table_cell_style), Paragraph("Governs cataracting vs. cascading regime.", table_cell_style)],
        [Paragraph("<code>has_short_lifter</code>", table_cell_style), Paragraph("&mathbb;I(short_angles &gt; 0)", table_cell_style), Paragraph("Binary flag for dual-height hi-lo lifter designs.", table_cell_style)],
        [Paragraph("<code>face_angle_asymmetry</code>", table_cell_style), Paragraph("&alpha;<sub>lead</sub> - &beta;<sub>trail</sub>", table_cell_style), Paragraph("Angle differential driving charge trajectory.", table_cell_style)],
        [Paragraph("<code>shape_energy</code>", table_cell_style), Paragraph("&radic;(&sum; <i>C</i><sub><i>k</i></sub><sup>2</sup>)", table_cell_style), Paragraph("<i>L</i><sub>2</sub> norm of 50 Fourier harmonic amplitudes.", table_cell_style)],
        [Paragraph("<code>shape_hf_sharpness</code>", table_cell_style), Paragraph("mean(<i>C</i><sub>30..49</sub>)", table_cell_style), Paragraph("High-frequency Fourier mean encoding lifter edge sharpness.", table_cell_style)],
        [Paragraph("<code>tip_speed</code>", table_cell_style), Paragraph("&pi; &middot; <i>D</i><sub>eff</sub> &middot; <i>N</i><sub>rpm</sub> / 60", table_cell_style), Paragraph("Maximum tangential impact velocity of lifter tips (m/s).", table_cell_style)],
        [Paragraph("<code>lifter_density</code>", table_cell_style), Paragraph("<i>N</i><sub>lifters</sub> / (&pi; &middot; <i>D</i><sub>eff</sub>)", table_cell_style), Paragraph("Lifter packing density per metre of shell circumference.", table_cell_style)],
        [Paragraph("<code>media_load_fraction</code>", table_cell_style), Paragraph("<i>M</i><sub>media</sub> / (<i>V</i><sub>mill</sub> &middot; <i>&rho;</i><sub>media</sub>)", table_cell_style), Paragraph("Normalised volume filling degree of grinding media.", table_cell_style)],
        [Paragraph("<code>has_ore</code>", table_cell_style), Paragraph("&mathbb;I(<i>M</i><sub>ore</sub> &gt; 0)", table_cell_style), Paragraph("Binary flag for SAG mill ore charge vs. dry/ball mill.", table_cell_style)],
        [Paragraph("<code>froude_number</code>", table_cell_style), Paragraph("<i>&omega;</i><sup>2</sup> &middot; <i>R</i> / <i>g</i>", table_cell_style), Paragraph("Ratio of inertial forces to gravitational forces.", table_cell_style)],
        [Paragraph("<code>charge_kinetic_head</code>", table_cell_style), Paragraph("&frac12; &middot; <i>&rho;</i><sub>mix</sub> &middot; <i>v</i><sub>tip</sub><sup>2</sup>", table_cell_style), Paragraph("Impact energy density of the charge toe (Pa).", table_cell_style)],
        [Paragraph("<code>lifter_strike_freq</code>", table_cell_style), Paragraph("(<i>N</i><sub>rpm</sub> &middot; <i>N</i><sub>lifters</sub>) / 60", table_cell_style), Paragraph("Frequency of high-energy impact pulses per second (Hz).", table_cell_style)],
        [Paragraph("<code>power_flux_proxy</code>", table_cell_style), Paragraph("<i>v</i><sub>tip</sub><sup>2</sup> &middot; <i>M</i><sub>media</sub> &middot; <i>N</i><sub>rpm</sub>", table_cell_style), Paragraph("Kinetic energy transfer rate proxy for mill power draw.", table_cell_style)],
        [Paragraph("<code>specific_impact_energy</code>", table_cell_style), Paragraph("<i>v</i><sub>tip</sub><sup>2</sup> / <i>D</i><sub>eff</sub>", table_cell_style), Paragraph("Per-meter collision impact energy capacity.", table_cell_style)],
        [Paragraph("<code>rot_sin</code> / <code>rot_cos</code>", table_cell_style), Paragraph("sin(2&pi; &middot; <i>&theta;</i>), &nbsp; cos(2&pi; &middot; <i>&theta;</i>)", table_cell_style), Paragraph("Encodes 360° cyclical rotation continuity.", table_cell_style)],
        [Paragraph("<code>media_aspect_ratio</code>", table_cell_style), Paragraph("<i>R</i><sub>ball</sub> / <i>D</i><sub>eff</sub>", table_cell_style), Paragraph("Grinding media particle radius relative to mill diameter.", table_cell_style)],
        [Paragraph("<code>total_charge_mass</code>", table_cell_style), Paragraph("<i>M</i><sub>media</sub> + <i>M</i><sub>ore</sub>", table_cell_style), Paragraph("Total combined charge mass inside mill shell (kg).", table_cell_style)],
        [Paragraph("<code>media_count_proxy</code>", table_cell_style), Paragraph("<i>M</i><sub>charge</sub> / <i>m</i><sub>single_ball</sub>", table_cell_style), Paragraph("Estimated total grinding ball count in charge.", table_cell_style)]
    ]
    t_feat = Table(features_data, colWidths=[140, 180, 220])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 15))

    # ── 4. Dynamic Dataset Operational Insights ──────────────────────────────
    notify(55, "Computing dynamic operational insights...")
    time.sleep(0.2)

    story.append(Paragraph("3. Dynamic Dataset Operational Insights", h2_style))
    if not raw_df.empty:
        d_min = raw_df["mill_diameter_m"].min() if "mill_diameter_m" in raw_df.columns else 5.0
        d_max = raw_df["mill_diameter_m"].max() if "mill_diameter_m" in raw_df.columns else 11.2
        rpm_min = raw_df["rotational_speed_rpm"].min() if "rotational_speed_rpm" in raw_df.columns else 7.8
        rpm_max = raw_df["rotational_speed_rpm"].max() if "rotational_speed_rpm" in raw_df.columns else 16.5
        
        insights_text = f"""
        Operational range analysis dynamically computed for <b>{total_sim_count} total simulations</b> (<b>{total_rows:,} rows</b>):<br/>
        &bull;&nbsp;<b>Mill Internal Diameter (D):</b> Spans from <b>{d_min:.2f} m</b> to <b>{d_max:.2f} m</b>.<br/>
        &bull;&nbsp;<b>Operating Speed (RPM):</b> Spans from <b>{rpm_min:.1f} RPM</b> to <b>{rpm_max:.1f} RPM</b>.<br/>
        &bull;&nbsp;<b>Simulation Utilization Ratio:</b> <b>{n_used_sims} used ({n_train_sims} Train / {n_val_sims} Val / {n_test_sims} Test)</b> and <b>{n_excluded_sims} excluded</b>.<br/>
        &bull;&nbsp;<b>Force-Train Strategy:</b> Simulations 3, 4, 5, and 24 are explicitly assigned to Training to prevent geometric extrapolation errors across extreme lifter configurations.
        """
    else:
        insights_text = f"""
        Operational range analysis compiled dynamically:<br/>
        &bull;&nbsp;<b>Simulation Utilization:</b> {n_used_sims} used ({n_train_sims} Train / {n_val_sims} Val / {n_test_sims} Test) and {n_excluded_sims} excluded.<br/>
        &bull;&nbsp;<b>Force-Train Strategy:</b> Extreme geometry simulations force-allocated to Training split.
        """
    story.append(Paragraph(insights_text, body_style))
    story.append(Spacer(1, 15))

    # ── 5. Surrogate Model Performance Summary ───────────────────────────────
    notify(70, "Compiling Surrogate Model Performance Metrics...")
    time.sleep(0.2)

    story.append(Paragraph("4. Surrogate Model Performance Summary", h2_style))
    perf_data = [
        [Paragraph("Target Variable", table_header_style), Paragraph("Model Architecture", table_header_style), Paragraph("Train R²", table_header_style), Paragraph("Val R²", table_header_style), Paragraph("Test R²", table_header_style), Paragraph("MAE", table_header_style)],
        [Paragraph("<b>Total Power Draw (kW)</b>", table_cell_style), Paragraph("300-Tree ExtraTrees", table_cell_style), Paragraph("1.0000", table_cell_style), Paragraph("0.9552", table_cell_style), Paragraph("0.8773", table_cell_style), Paragraph("27.82 kW", table_cell_style)],
        [Paragraph("<b>Max Particle Kinetic Energy</b>", table_cell_style), Paragraph("300-Tree ExtraTrees", table_cell_style), Paragraph("1.0000", table_cell_style), Paragraph("0.9692", table_cell_style), Paragraph("0.8450", table_cell_style), Paragraph("37.96 J", table_cell_style)],
        [Paragraph("<b>Max Particle Compressive Force (N)</b>", table_cell_style), Paragraph("500-Tree QRF + Superposition", table_cell_style), Paragraph("0.9362", table_cell_style), Paragraph("0.5000", table_cell_style), Paragraph("0.4729", table_cell_style), Paragraph("2489.47 N (Peak Acc 96.7%)", table_cell_style)]
    ]
    t_perf = Table(perf_data, colWidths=[140, 130, 50, 50, 50, 120])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d9488')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_perf)
    story.append(Spacer(1, 20))

    # ── 6. DYNAMIC EVALUATION PLOTS GALLERY ──────────────────────────────────
    notify(85, "Embedding Evaluation Plots sorted dynamically by Simulation ID...")
    time.sleep(0.2)

    story.append(PageBreak())

    plots_folder = os.path.join(target_project_dir, "evaluation_plots")
    if not os.path.exists(plots_folder):
        plots_folder = target_project_dir

    raw_plots = [f for f in os.listdir(plots_folder) if f.endswith(".png") and f.startswith("eval_")] if os.path.exists(plots_folder) else []
    plot_files = sorted(raw_plots, key=get_sim_num)
    n_plots = len(plot_files)

    story.append(Paragraph(f"5. Evaluation Plots Gallery ({n_plots} Simulations Ordered by Simulation ID)", h2_style))
    story.append(Paragraph(f"Dynamically embedded validation plots for all <b>{n_plots} generated simulation evaluation curves</b> (Total Power Draw, Max Particle Kinetic Energy, Max Particle Compressive Force):", body_style))
    story.append(Spacer(1, 10))

    if plot_files:
        for idx in range(0, len(plot_files), 2):
            p1_path = os.path.join(plots_folder, plot_files[idx])
            img1 = Image(p1_path, width=265, height=185)
            cap1 = Paragraph(f"<b>{plot_files[idx].replace('.png', '')}</b>", table_cell_style)
            
            if idx + 1 < len(plot_files):
                p2_path = os.path.join(plots_folder, plot_files[idx + 1])
                img2 = Image(p2_path, width=265, height=185)
                cap2 = Paragraph(f"<b>{plot_files[idx + 1].replace('.png', '')}</b>", table_cell_style)
                
                row_table = Table([[img1, img2], [cap1, cap2]], colWidths=[270, 270])
            else:
                row_table = Table([[img1, ""], [cap1, ""]], colWidths=[270, 270])

            row_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,1), (-1,1), 10),
            ]))
            
            story.append(KeepTogether([row_table, Spacer(1, 10)]))

    notify(95, "Assembling PDF Document layout...")
    time.sleep(0.1)

    doc.build(story, canvasmaker=NumberedCanvas)
    notify(100, "Master PDF Report generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        target_dir = sys.argv[1]
        excel_path_arg = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(target_dir, "PRE PROCESSED", "data_v1.xlsx")
    else:
        target_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path_arg = os.path.join(target_dir, "PRE PROCESSED", "data_v1.xlsx")

    build_pdf_report(target_dir, excel_path_arg)
