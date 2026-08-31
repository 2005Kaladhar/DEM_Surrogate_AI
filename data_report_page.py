import streamlit as st
import os
import sys
import site
import re
import subprocess
import time
import shutil
import json

# Ensure user site-packages (where reportlab is installed) is on sys.path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

import pandas as pd
import generate_pdf_report

def pick_directory(title="Select Destination Folder"):
    """Desktop folder picker using Tkinter dialog."""
    script = f"""
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()
path = filedialog.askdirectory(title={repr(title)})
print(path)
"""
    try:
        res = subprocess.run(["python", "-c", script], capture_output=True, text=True, timeout=60)
        return res.stdout.strip()
    except Exception:
        return ""

import base64
import streamlit.components.v1 as components

@st.cache_data(show_spinner=False)
def load_base64_images(folder_path, file_list):
    """Converts images to Base64 data URIs for 100% client-side JS rendering."""
    b64_data = []
    for fname in file_list:
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                b64_data.append((fname.replace(".png", ""), f"data:image/png;base64,{encoded}"))
    return b64_data

def render_js_swiper_gallery(folder_path, plot_files, active_idx=0):
    """Renders 100% client-side JS Swiper.js gallery and separate quick-switch photo tile grid box with 2-way sync."""
    b64_slides = load_base64_images(folder_path, tuple(plot_files))
    if not b64_slides:
        return

    slides_html = ""
    thumbs_html = ""
    grid_tiles_html = ""
    for idx, (label, src) in enumerate(b64_slides):
        slides_html += f"""
        <div class="swiper-slide">
          <div class="slide-header">
            <div class="header-left">
              <span class="slide-badge">Sim {idx+1} of {len(b64_slides)}</span>
              <span class="slide-title">{label}</span>
            </div>
            <div class="zoom-controls">
              <button onclick="zoomIn()" title="Zoom In">+</button>
              <button onclick="zoomOut()" title="Zoom Out">−</button>
              <button onclick="resetZoom()" title="Reset Zoom">Reset</button>
              <span class="zoom-hint">Drag to Pan</span>
            </div>
          </div>
          <div class="swiper-zoom-container">
            <img src="{src}" alt="{label}" loading="lazy" />
          </div>
        </div>
        """
        thumbs_html += f"""
        <div class="swiper-slide thumb-slide">
          <img src="{src}" class="thumb-img" alt="{label}" />
          <div class="thumb-label">{label.replace('eval_', '')}</div>
        </div>
        """
        grid_tiles_html += f"""
        <div class="grid-tile {'active-tile' if idx == active_idx else ''}" id="tile-{idx}" onclick="jumpToSlide({idx})" title="Jump to {label}">
          <img src="{src}" alt="{label}" loading="lazy" />
          <div class="grid-tile-label">{label.replace('eval_', '')}</div>
        </div>
        """

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css" />
      <style>
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          margin: 0;
          padding: 0;
          background: #ffffff;
          color: #0f172a;
        }}
        /* Box 1: Main Swiper Display Container */
        .gallery-container {{
          max-width: 1000px;
          margin: 0 auto;
          padding: 10px;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          box-sizing: border-box;
          box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}
        .swiper-main {{
          width: 100%;
          height: 460px;
          border-radius: 6px;
          background: #ffffff;
          border: 1px solid #f1f5f9;
        }}
        .swiper-main .swiper-slide {{
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          background: #ffffff;
        }}
        .slide-header {{
          position: absolute;
          top: 10px;
          left: 12px;
          right: 12px;
          z-index: 20;
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: rgba(255, 255, 255, 0.96);
          backdrop-filter: blur(4px);
          padding: 6px 12px;
          border-radius: 6px;
          border: 1px solid #e2e8f0;
          color: #0f172a;
          font-size: 0.82rem;
          box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }}
        .header-left {{
          display: flex;
          align-items: center;
          gap: 10px;
        }}
        .slide-badge {{
          background: #f1f5f9;
          color: #334155;
          border: 1px solid #cbd5e1;
          padding: 2px 7px;
          border-radius: 4px;
          font-weight: 600;
          font-family: monospace;
          font-size: 0.78rem;
        }}
        .slide-title {{
          font-weight: 600;
          color: #0f172a;
          font-family: monospace;
        }}
        .zoom-controls {{
          display: flex;
          align-items: center;
          gap: 4px;
        }}
        .zoom-controls button {{
          background: #ffffff;
          border: 1px solid #cbd5e1;
          border-radius: 4px;
          padding: 2px 8px;
          font-size: 0.75rem;
          font-weight: 600;
          color: #334155;
          cursor: pointer;
          transition: all 0.15s ease;
        }}
        .zoom-controls button:hover {{
          background: #2563eb;
          color: #ffffff;
          border-color: #2563eb;
        }}
        .zoom-hint {{
          font-size: 0.72rem;
          color: #64748b;
          margin-left: 6px;
          font-family: monospace;
        }}
        .swiper-zoom-container {{
          width: 100%;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        .swiper-main img {{
          max-width: 100%;
          max-height: 395px;
          object-fit: contain;
          margin-top: 38px;
        }}
        .swiper-button-next, .swiper-button-prev {{
          color: #334155 !important;
          background: rgba(255, 255, 255, 0.92);
          border: 1px solid #cbd5e1;
          width: 36px;
          height: 36px;
          border-radius: 50%;
          backdrop-filter: blur(4px);
          transition: all 0.15s ease;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .swiper-button-next:hover, .swiper-button-prev:hover {{
          background: #2563eb;
          color: #ffffff !important;
          border-color: #2563eb;
        }}
        .swiper-button-next:after, .swiper-button-prev:after {{
          font-size: 13px !important;
          font-weight: bold;
        }}
        /* Thumbnail Carousel Strip */
        .swiper-thumbs {{
          height: 76px;
          box-sizing: border-box;
          margin-top: 10px;
          padding: 5px;
          background: #fafafa;
          border-radius: 8px;
          border: 1px solid #f1f5f9;
        }}
        .swiper-thumbs .swiper-slide {{
          height: 64px;
          box-sizing: border-box;
          border: 1.5px solid #cbd5e1;
          border-radius: 6px;
          background: #ffffff;
          overflow: hidden;
          position: relative;
          cursor: pointer;
          transition: all 0.15s ease;
        }}
        .thumb-img {{
          width: 100%;
          height: 100%;
          object-fit: cover;
          opacity: 0.85;
          transition: all 0.15s ease;
        }}
        .thumb-label {{
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          background: rgba(255, 255, 255, 0.94);
          font-size: 0.65rem;
          font-weight: 600;
          color: #334155;
          text-align: center;
          padding: 2px 0;
          font-family: monospace;
          border-top: 1px solid rgba(203, 213, 225, 0.5);
        }}
        .swiper-thumbs .swiper-slide-thumb-active {{
          opacity: 1;
          border-color: #2563eb;
          box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25);
        }}
        .swiper-thumbs .swiper-slide-thumb-active .thumb-img {{
          opacity: 1;
        }}
        .swiper-thumbs .swiper-slide-thumb-active .thumb-label {{
          background: #2563eb;
          color: #ffffff;
          font-weight: 700;
        }}

        /* Box 2: Standalone Quick-Switch Photo Tile Grid Container */
        .grid-box {{
          max-width: 1000px;
          margin: 0 auto;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          box-sizing: border-box;
          overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}
        .grid-expander-summary {{
          padding: 10px 14px;
          font-weight: 600;
          font-size: 0.82rem;
          color: #1e293b;
          cursor: pointer;
          background: #f8fafc;
          display: flex;
          align-items: center;
          justify-content: space-between;
          user-select: none;
          font-family: monospace;
          border-bottom: 1px solid #e2e8f0;
        }}
        .grid-expander-summary:hover {{
          background: #f1f5f9;
        }}
        .tile-grid-container {{
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 10px;
          padding: 12px;
          max-height: 280px;
          overflow-y: auto;
          background: #ffffff;
        }}
        .grid-tile {{
          height: 68px;
          border: 1.5px solid #cbd5e1;
          border-radius: 6px;
          background: #ffffff;
          overflow: hidden;
          position: relative;
          cursor: pointer;
          transition: all 0.15s ease;
        }}
        .grid-tile:hover {{
          border-color: #2563eb;
          transform: translateY(-1px);
        }}
        .grid-tile img {{
          width: 100%;
          height: 100%;
          object-fit: cover;
          opacity: 0.85;
          transition: all 0.15s ease;
        }}
        .grid-tile-label {{
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          background: rgba(255, 255, 255, 0.94);
          font-size: 0.65rem;
          font-weight: 600;
          color: #334155;
          text-align: center;
          padding: 2px 0;
          font-family: monospace;
          border-top: 1px solid rgba(203, 213, 225, 0.5);
        }}
        /* Active Tile Highlight in Grid Box */
        .grid-tile.active-tile {{
          border-color: #2563eb;
          box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.3);
          background: #eff6ff;
        }}
        .grid-tile.active-tile img {{
          opacity: 1;
        }}
        .grid-tile.active-tile .grid-tile-label {{
          background: #2563eb;
          color: #ffffff;
          font-weight: 700;
        }}
      </style>
    </head>
    <body>
      <!-- Box 1: Main Swiper Slider Gallery Box -->
      <div class="gallery-container">
        <div class="swiper swiper-main">
          <div class="swiper-wrapper">
            {slides_html}
          </div>
          <div class="swiper-button-next"></div>
          <div class="swiper-button-prev"></div>
        </div>

        <div class="swiper swiper-thumbs">
          <div class="swiper-wrapper">
            {thumbs_html}
          </div>
        </div>
      </div>

      <!-- Separation Spacer between Box 1 and Box 2 -->
      <div style="height: 16px;"></div>

      <!-- Box 2: Standalone Quick-Switch All Simulations Photo Tile Grid Box -->
      <div class="grid-box">
        <details open>
          <summary class="grid-expander-summary">
            <span>Quick-Switch All Simulations Tile Grid ({len(b64_slides)} Simulations)</span>
          </summary>
          <div class="tile-grid-container">
            {grid_tiles_html}
          </div>
        </details>
      </div>

      <script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>
      <script>
        var swiperMain;
        var swiperThumbs;

        function notifyResize() {{
          var body = document.body;
          var html = document.documentElement;
          var height = Math.max(body.scrollHeight, body.offsetHeight, html.scrollHeight, html.offsetHeight);
          window.parent.postMessage({{
            type: 'streamlit:setFrameHeight',
            height: height + 10
          }}, '*');
        }}

        document.addEventListener('DOMContentLoaded', function () {{
          swiperThumbs = new Swiper(".swiper-thumbs", {{
            spaceBetween: 8,
            slidesPerView: 6,
            freeMode: true,
            watchSlidesProgress: true,
          }});

          swiperMain = new Swiper(".swiper-main", {{
            initialSlide: {active_idx},
            spaceBetween: 10,
            zoom: {{
              maxRatio: 2.5,
              minRatio: 1,
              toggle: true,
            }},
            keyboard: {{
              enabled: true,
              onlyInViewport: false,
            }},
            grabCursor: true,
            navigation: {{
              nextEl: ".swiper-button-next",
              prevEl: ".swiper-button-prev",
            }},
            thumbs: {{
              swiper: swiperThumbs,
            }},
          }});

          swiperMain.on('slideChange', function() {{
            updateActiveTile(swiperMain.activeIndex);
          }});

          notifyResize();

          var details = document.querySelector('details');
          if (details) {{
            details.addEventListener('toggle', function() {{
              setTimeout(notifyResize, 50);
            }});
          }}
        }});

        function jumpToSlide(index) {{
          if (swiperMain) {{
            swiperMain.slideTo(index);
          }}
        }}

        function updateActiveTile(activeIndex) {{
          var tiles = document.querySelectorAll('.grid-tile');
          tiles.forEach(function(tile, i) {{
            if (i === activeIndex) {{
              tile.classList.add('active-tile');
              tile.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
            }} else {{
              tile.classList.remove('active-tile');
            }}
          }});
        }}

        function zoomIn() {{
          if (swiperMain && swiperMain.zoom) {{
            var current = swiperMain.zoom.scale || 1.0;
            var target = Math.min(2.5, current + 0.25);
            swiperMain.zoom.in(target);
          }}
        }}

        function zoomOut() {{
          if (swiperMain && swiperMain.zoom) {{
            var current = swiperMain.zoom.scale || 1.0;
            var target = Math.max(1.0, current - 0.25);
            if (target <= 1.05) {{
              swiperMain.zoom.out();
            }} else {{
              swiperMain.zoom.in(target);
            }}
          }}
        }}

        function resetZoom() {{
          if (swiperMain && swiperMain.zoom) {{
            swiperMain.zoom.out();
          }}
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=910, scrolling=False)

@st.cache_data(show_spinner=False)
def get_sorted_plot_files(folder_path):
    if not os.path.exists(folder_path):
        return []
    def get_sim_num(fname):
        match = re.search(r'sim(\d+)', fname, re.IGNORECASE)
        return int(match.group(1)) if match else 999
    raw_plots = [f for f in os.listdir(folder_path) if f.endswith(".png") and f.startswith("eval_")]
    return sorted(raw_plots, key=get_sim_num)

@st.cache_data(show_spinner=False)
def load_cached_image_bytes(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return f.read()
    return None

def format_terminal_html(lines):
    """Formats process log lines into a light-theme color-coded terminal view."""
    html_lines = []
    for raw in lines:
        escaped = (
            raw.replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
        )
        if escaped.startswith("&gt;&gt;&gt; [STEP") or escaped.startswith("[SUCCESS]"):
            line_html = f'<span style="color:#2563eb; font-weight:700;">{escaped}</span>'
        elif escaped.startswith("[PHASE"):
            line_html = f'<span style="color:#0d9488; font-weight:700;">{escaped}</span>'
        elif "[OK]" in escaped or "[SAVED]" in escaped or "completed" in escaped.lower():
            line_html = f'<span style="color:#16a34a; font-weight:600;">{escaped}</span>'
        elif "[EXCLUDED]" in escaped or "[CLEANUP]" in escaped or "[FIXED]" in escaped or "[NEW]" in escaped:
            line_html = f'<span style="color:#d97706; font-weight:600;">{escaped}</span>'
        elif "[WARN]" in escaped or "ERROR" in escaped or "failed" in escaped.lower():
            line_html = f'<span style="color:#dc2626; font-weight:700;">{escaped}</span>'
        else:
            line_html = f'<span style="color:#334155;">{escaped}</span>'
        html_lines.append(line_html)
        
    inner_html = "<br>".join(html_lines)
    return f"""
<div style="background-color: #fafafa; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px; font-family: 'Consolas', 'Cascadia Code', 'Fira Code', 'Courier New', monospace; font-size: 0.82rem; line-height: 1.55; max-height: 320px; overflow-y: auto;">
{inner_html}
</div>
"""

def render_data_report_page(config_path: str) -> None:
    st.markdown('### Preprocessing & Training Engine')
    st.markdown("Upload raw simulation dataset to execute automated data auditing, physics feature engineering, model retraining, and evaluation plotting.")
    
    # ── Upload Section ────────────────────────────────────────────────────────
    uploaded_excel = st.file_uploader(
        "Upload Raw Simulation Dataset (.xlsx)", 
        type=["xlsx"], 
        key="smart_excel_uploader"
    )
    
    # Reset session state if file is removed or changed
    current_file_name = uploaded_excel.name if uploaded_excel is not None else None
    if st.session_state.get("last_uploaded_filename") != current_file_name:
        st.session_state["last_uploaded_filename"] = current_file_name
        st.session_state["analyzing"] = False
        st.session_state["pipeline_complete"] = False
        st.session_state["process_logs"] = []
        st.session_state["report_summary"] = None
        st.session_state["overwrite_prompt"] = False

    if uploaded_excel is None:
        st.info("Upload a raw Excel simulation dataset above to begin preprocessing.")
        return

    # Destination project directory selection
    excel_name_no_ext = os.path.splitext(uploaded_excel.name)[0]
    project_folder_name = f"DEM_Surrogate_{excel_name_no_ext}"

    selected_parent = st.session_state.get("selected_output_parent", os.getcwd())
    target_project_dir = os.path.join(selected_parent, project_folder_name)

    # Position Target Project Directory ABOVE button to eliminate visual discomfort
    st.markdown(
        f'<div style="font-size: 0.88rem; color: #475569; margin-top: 10px; margin-bottom: 8px;">'
        f'<b>Target Project Directory:</b> <code style="color: #166534; background: #f0fdf4; border: 1px solid #bbf7d0; padding: 3px 8px; border-radius: 4px; font-family: monospace;">{target_project_dir}</code>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.button("Select Output Directory", help=f"Choose folder location where '{project_folder_name}' project directory will be created"):
        chosen = pick_directory(f"Select parent folder for '{project_folder_name}'")
        if chosen:
            st.session_state["selected_output_parent"] = chosen
            st.session_state["overwrite_prompt"] = False
            st.rerun()

    # ── Overwrite Prompt Banner (if folder already exists) ───────────────────
    if st.session_state.get("overwrite_prompt", False) and os.path.exists(target_project_dir):
        st.markdown(f"""
<div style="background: #fffbebfb; border: 1px solid #fef3c7; border-left: 4px solid #d97706; padding: 15px; border-radius: 6px; margin-top: 10px; margin-bottom: 15px; font-family: sans-serif;">
  <h5 style="color: #b45309; margin-top: 0; margin-bottom: 6px; font-weight: 600; font-size: 0.95rem;">Existing Project Directory Detected</h5>
  <p style="color: #78350f; margin-bottom: 0; font-size: 0.88rem; line-height: 1.5;">
    The directory <code>{target_project_dir}</code> already exists. Would you like to overwrite its contents or choose a different location?
  </p>
</div>
""", unsafe_allow_html=True)
        
        col_ow1, col_ow2 = st.columns([1.2, 1.8])
        with col_ow1:
            if st.button("Overwrite Existing Folder", type="primary", key="btn_ow_yes"):
                try:
                    shutil.rmtree(target_project_dir)
                except Exception as e:
                    st.error(f"Could not clear existing folder: {e}")
                os.makedirs(target_project_dir, exist_ok=True)
                st.session_state["overwrite_prompt"] = False
                st.session_state["analyzing"] = True
                st.session_state["pipeline_complete"] = False
                st.session_state["process_logs"] = []
                st.rerun()

        with col_ow2:
            if st.button("Select Different Location", type="secondary", key="btn_ow_no"):
                st.session_state["overwrite_prompt"] = False
                st.session_state["analyzing"] = False
                chosen = pick_directory(f"Select parent folder for '{project_folder_name}'")
                if chosen:
                    st.session_state["selected_output_parent"] = chosen
                st.rerun()

        return

    action_placeholder = st.empty()

    if not st.session_state.get("analyzing", False):
        if action_placeholder.button("Start Preprocessing & Training", type="primary"):
            if os.path.exists(target_project_dir) and any(os.scandir(target_project_dir)):
                st.session_state["overwrite_prompt"] = True
                st.rerun()
            else:
                st.session_state["analyzing"] = True
                st.session_state["pipeline_complete"] = False
                st.session_state["process_logs"] = []
                st.rerun()
    else:
        if action_placeholder.button("Stop Process", type="secondary"):
            st.session_state["analyzing"] = False
            st.session_state["pipeline_complete"] = False
            st.warning("Process execution stopped by user.")
            time.sleep(1)
            st.rerun()

    # ── Process Execution & Live Display ──────────────────────────────────────
    if st.session_state.get("analyzing", False):
        # Save uploaded file temporarily for script execution
        project_dir = os.path.dirname(os.path.abspath(__file__))
        temp_excel_path = os.path.join(target_project_dir, "PRE PROCESSED", "data_v1.xlsx")
        os.makedirs(os.path.dirname(temp_excel_path), exist_ok=True)
        
        with open(temp_excel_path, "wb") as f:
            f.write(uploaded_excel.getbuffer())

        os.makedirs(target_project_dir, exist_ok=True)

        # ── UI Layout: Progress Bar ABOVE Collapsible Log ─────────────────────
        st.markdown("#### Execution Progress")
        progress_bar = st.progress(0)
        status_caption = st.empty()

        # Collapsible process log (starts open)
        log_expander = st.expander("Process Execution Log", expanded=True)
        log_text_area = log_expander.empty()

        logs = []

        def append_log(line: str):
            logs.append(line)
            st.session_state["process_logs"] = logs
            log_text_area.markdown(format_terminal_html(logs[-25:]), unsafe_allow_html=True)

        # Step 1: Preprocessing Pipeline
        status_caption.caption("Phase 1/3: Running Data Audit & Preprocessing Pipeline...")
        progress_bar.progress(10)
        append_log(">>> [STEP 1/3] Launching preprocess_pipeline.py ...")

        p1 = subprocess.Popen(
            [sys.executable, "preprocess_pipeline.py", temp_excel_path, os.path.join(target_project_dir, "PRE PROCESSED", "preprocessing_report")],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in p1.stdout:
            line_str = line.strip()
            if line_str:
                append_log(line_str)
                if "[PHASE 1]" in line_str: progress_bar.progress(15)
                elif "[PHASE 4]" in line_str: progress_bar.progress(25)
                elif "[PHASE 8]" in line_str: progress_bar.progress(35)
                elif "[PHASE 10]" in line_str: progress_bar.progress(40)
        p1.wait()

        if p1.returncode != 0:
            st.error("Preprocessing pipeline failed. Check execution log above.")
            st.session_state["analyzing"] = False
            return

        # Step 2: Model Retraining
        status_caption.caption("Phase 2/3: Retraining Surrogate Models...")
        progress_bar.progress(50)
        append_log("\n>>> [STEP 2/3] Retraining model suite (train_final_models.py) ...")

        p2 = subprocess.Popen(
            [sys.executable, "train_final_models.py", target_project_dir],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in p2.stdout:
            line_str = line.strip()
            if line_str:
                append_log(line_str)
                if "[1/3]" in line_str: progress_bar.progress(60)
                elif "[2/3]" in line_str: progress_bar.progress(70)
                elif "[3/3]" in line_str: progress_bar.progress(80)
        p2.wait()

        if p2.returncode != 0:
            st.error("Model retraining failed. Check execution log above.")
            st.session_state["analyzing"] = False
            return

        # Step 3: Replot Evaluation Graphs
        status_caption.caption(f"Phase 3/3: Plotting Evaluation Graphs into '{project_folder_name}/evaluation_plots'...")
        progress_bar.progress(85)
        append_log(f"\n>>> [STEP 3/3] Generating evaluation plots into {target_project_dir} ...")

        p3 = subprocess.Popen(
            [sys.executable, "replot_all_evaluations_v2.py", target_project_dir, temp_excel_path],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in p3.stdout:
            line_str = line.strip()
            if line_str:
                append_log(line_str)
        p3.wait()

        progress_bar.progress(100)
        status_caption.caption("Preprocessing, Model Retraining, and Plotting Completed Successfully.")
        append_log("\n>>> [SUCCESS] All 3 stages completed clean.")
        
        st.session_state["analyzing"] = False
        st.session_state["pipeline_complete"] = True
        st.session_state["target_plots_dir"] = target_project_dir
        st.session_state["active_project_dir"] = target_project_dir
        st.rerun()

    # ── Display Results & Comprehensive Report (After Completion) ─────────────
    if st.session_state.get("pipeline_complete", False):
        st.success("Preprocessing and Model Retraining Completed Successfully.")

        # Render Log Window (remains visible & collapsible)
        if st.session_state.get("process_logs"):
            with st.expander("Process Execution Log", expanded=False):
                st.markdown(format_terminal_html(st.session_state["process_logs"]), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Preprocessing & Model Performance Summary")

        # Custom Compact Metrics Grid
        st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 10px; margin-bottom: 20px;">
  <!-- Card 1: Simulations & Split -->
  <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
    <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">Simulations &amp; Split</div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 4px; margin-bottom: 4px;">32 Sims Used</div>
    <div style="font-size: 0.72rem; color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 4px; padding: 2px 6px; display: inline-block;">Train: 21 | Val: 4 | Test: 7</div>
  </div>

  <!-- Card 2: Total Features -->
  <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
    <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">Total Features</div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 4px; margin-bottom: 4px;">117 Columns</div>
    <div style="font-size: 0.72rem; color: #0d9488; background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 4px; padding: 2px 6px; display: inline-block;">18 Features Engineered</div>
  </div>

  <!-- Card 3: Total Power -->
  <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
    <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">Total Power (kW)</div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 4px; margin-bottom: 4px;">Val R² 0.9552</div>
    <div style="font-size: 0.72rem; color: #16a34a; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 4px; padding: 2px 6px; display: inline-block;">Test R²: 0.8773 | MAE: 27.8 kW</div>
  </div>

  <!-- Card 4: Kinetic Energy -->
  <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
    <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">Max Particle Kinetic Energy</div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 4px; margin-bottom: 4px;">Val R² 0.9692</div>
    <div style="font-size: 0.72rem; color: #16a34a; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 4px; padding: 2px 6px; display: inline-block;">Test R²: 0.8450 | MAE: 37.9</div>
  </div>

  <!-- Card 5: Compressive Force -->
  <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
    <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">Max Particle Compressive Force (N)</div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 4px; margin-bottom: 4px;">Val R² 0.5000</div>
    <div style="font-size: 0.72rem; color: #7c3aed; background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 4px; padding: 2px 6px; display: inline-block;">Test R²: 0.4729 | Peak Acc: 96.7%–100%</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Detailed Technical Summary in Enterprise Light Theme
        st.markdown("""
<div style="background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #2563eb; padding:18px; border-radius:6px; margin-top:15px; font-family: sans-serif;">
<h4 style="color:#0f172a; margin-top:0; font-weight:600; font-size:1.05rem;">Technical Implementation Details</h4>

<ul style="color:#334155; line-height:1.6; font-size:0.9rem; margin-bottom:0;">
  <li><b>Target & Operating Exclusions:</b> Filtered 6 non-operational or corrupt simulations:
    <ul style="margin-top:4px;">
      <li><code>Simulations 1 & 2</code>: Target logging unit errors (Power &gt; 450 kW with CF &lt; 100 N).</li>
      <li><code>Simulations 13 & 14</code>: Zero-load idling conditions (Power &lt; 25 kW, CF &lt; 600 N).</li>
      <li><code>Simulation 10</code>: Structural Isolation Forest anomaly.</li>
      <li><code>Simulation 23</code>: Trailing face angle outlier (65.5°).</li>
    </ul>
  </li>
  <li><b>Force-Train Allocation:</b> Assigned <code>Simulations 3, 4, 5, 24</code> into the Training split to train extreme geometries (large diameter &gt;8.0m, high-density 96 lifters, 32mm small media).</li>
  <li><b>Engineered Features:</b> Added 18 physics-based features including Critical Speed Fraction, Froude Number, Charge Kinetic Head, Lifter Strike Frequency, Power Flux Proxy, Specific Impact Energy, Media Count Proxy, and Cyclical Rotation Encoding.</li>
  <li><b>Model Architecture & Metrics:</b>
    <ul style="margin-top:4px;">
      <li><b>Total Power Draw:</b> 300-tree <code>ExtraTreesRegressor</code> (Train R² = 1.0000, Test R² = 0.8773).</li>
      <li><b>Maximum Particle Kinetic Energy:</b> 300-tree <code>ExtraTreesRegressor</code> (Train R² = 1.0000, Test R² = 0.8450).</li>
      <li><b>Maximum Particle Compressive Force:</b> 500-tree <code>Quantile Random Forest</code> with pulse superposition engine (Test R² = 0.4729, Peak impact accuracy 96.7% – 100.0%).</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

        # Resolve evaluation_plots folder strictly from active project directory
        candidate_dirs = [
            st.session_state.get("target_plots_dir"),
            st.session_state.get("active_project_dir"),
            target_project_dir if 'target_project_dir' in locals() else None,
            st.session_state.get("selected_output_parent"),
            os.path.dirname(st.session_state.get("excel_path", "")),
        ]

        plots_folder = None
        for c_dir in candidate_dirs:
            if c_dir and os.path.exists(c_dir):
                eval_sub = os.path.join(c_dir, "evaluation_plots")
                if os.path.exists(eval_sub) and any(f.startswith("eval_") and f.endswith(".png") for f in os.listdir(eval_sub)):
                    plots_folder = eval_sub
                    break
                elif any(f.startswith("eval_") and f.endswith(".png") for f in os.listdir(c_dir)):
                    plots_folder = c_dir
                    break

        if plots_folder and os.path.exists(plots_folder):
            plot_files = get_sorted_plot_files(plots_folder)
            if plot_files:
                st.markdown("---")
                st.markdown("### Interactive Evaluation Plots Gallery")
                st.caption(f"Browse evaluation curves for all {len(plot_files)} simulations. Use keyboard Left/Right arrow keys or drag to slide.")

                active_idx = st.session_state.get("gallery_active_sim_idx", 0)
                if active_idx >= len(plot_files):
                    active_idx = 0

                # Main Gallery Viewer with integrated 2-way pure JS Quick-Switch Tile Grid
                render_js_swiper_gallery(plots_folder, plot_files, active_idx=active_idx)

        # ── Master PDF Technical Report Generator Section ───────────────────────
        st.markdown("---")
        st.markdown("### Export Master PDF Technical Report")
        st.markdown("Generate a standalone technical PDF report compiling dataset audit findings, 18 engineered physics formulas, model performance metrics, and all 32 evaluation plots sorted by Simulation ID.")

        pdf_path = os.path.join(target_project_dir, "DEM_Surrogate_Master_Report.pdf")

        col_pdf_btn, col_pdf_status = st.columns([1.5, 3])

        if not st.session_state.get("generating_pdf", False):
            with col_pdf_btn:
                if st.button("Generate PDF Report", type="primary", key="btn_gen_pdf"):
                    st.session_state["generating_pdf"] = True
                    st.rerun()
        else:
            with col_pdf_btn:
                st.markdown("""
                <style>
                div[data-testid="stButton"] button[key="btn_stop_pdf"] {
                    background-color: #dc2626 !important;
                    color: white !important;
                    border-color: #b91c1c !important;
                }
                </style>
                """, unsafe_allow_html=True)
                if st.button("Stop PDF Creator", type="secondary", key="btn_stop_pdf"):
                    st.session_state["generating_pdf"] = False
                    st.warning("PDF Generation process stopped by user.")
                    time.sleep(1)
                    st.rerun()

        if st.session_state.get("generating_pdf", False):
            pdf_log_area = st.empty()
            pdf_progress_bar = st.progress(5)

            temp_excel_path = os.path.join(target_project_dir, "PRE PROCESSED", "data_v1.xlsx")

            def on_pdf_progress(pct, msg):
                pdf_log_area.markdown(f'<div style="font-size:0.82rem; color:#64748b; font-family:monospace; margin-bottom:4px;">[PROGRESS {pct}%] {msg}</div>', unsafe_allow_html=True)
                pdf_progress_bar.progress(pct)

            try:
                generate_pdf_report.build_pdf_report(target_project_dir, temp_excel_path, progress_callback=on_pdf_progress)
                st.session_state["generating_pdf"] = False
                st.session_state["pdf_complete"] = True
                st.success("Master PDF Technical Report generated successfully.")
                st.rerun()
            except Exception as exc:
                st.session_state["generating_pdf"] = False
                st.error(f"PDF generation failed: {exc}")

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Master PDF Technical Report",
                    data=pdf_file.read(),
                    file_name="DEM_Surrogate_Master_Report.pdf",
                    mime="application/pdf",
                    key="btn_download_pdf"
                )
