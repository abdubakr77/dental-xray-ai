"""
Analysis page: upload -> animated multi-stage inference -> final report.

This is a direct modularization of the existing app.py 'Try It' tab. The
animation mechanics (time.sleep + st.rerun, driven only by real pipeline
output already computed in st.session_state.result) are unchanged. What's
new here versus the original app.py:
  - pipeline-status stepper reflecting real stage completion
  - debug mode surfacing raw coordinates/shapes/timings
  - automatic report persistence via storage.reports.save_report
  - errors routed through core.errors instead of a bare st.error(str(e))
"""

import os
import io
import time
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference_pipeline import run_pipeline
from animation import inject_animation_css, render_image_with_overlay, render_tooth_grid, render_live_counter
from core.config import (
    CLASS_NAMES, STAGE_OPTIONS, STAGE_REQUIRED_INPUT, SAMPLE_IMAGES_DIR,
    TEMP_UPLOADS_DIR, DETECTION_CONF_THRESHOLD_DEFAULT,
)
from core.model_registry import get_models
from core.errors import AppError, show_error
from storage.reports import save_report
from ui.components import pipeline_status, disclaimer_banner, metric_row

STAGE_LABELS = ["Quadrants", "Teeth", "Health", "Disease", "Severity", "Summary"]

_DEFAULTS = {
    'result': None, 'warnings': [], 'running': False, 'stop_requested': False,
    'anim_stage': 0, 'anim_item': 0, 'selected_image_path': None,
    'selected_stage': 'full', 'run_history': [], 'run_started_at': None,
    'last_report_id': None,
}


def _init_state():
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_run():
    st.session_state.result = None
    st.session_state.warnings = []
    st.session_state.running = False
    st.session_state.stop_requested = False
    st.session_state.anim_stage = 0
    st.session_state.anim_item = 0
    st.session_state.last_report_id = None


def _image_to_base64(img_array) -> str:
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _list_sample_images(stage_key: str) -> list:
    if not os.path.isdir(SAMPLE_IMAGES_DIR):
        return []
    stage_folder = os.path.join(SAMPLE_IMAGES_DIR, stage_key)
    folder = stage_folder if os.path.isdir(stage_folder) else SAMPLE_IMAGES_DIR
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))


def _done_steps(anim_stage: int) -> list:
    return STAGE_LABELS[:anim_stage]


def render_sidebar_controls(debug: bool) -> tuple:
    """Returns (image_path, run_clicked, stop_clicked, reset_clicked)."""
    st.subheader("🦷 Run Analysis")

    stage_label = st.radio("**Which model to run?**", list(STAGE_OPTIONS.keys()),
                            help="Select where in the pipeline to start")
    stage_key = STAGE_OPTIONS[stage_label]
    st.session_state.selected_stage = stage_key

    with st.expander("📋 Input Requirements", expanded=True):
        st.info(f"**Expected input:** {STAGE_REQUIRED_INPUT[stage_key]}")

    st.divider()
    st.subheader("📸 Select Image")

    samples = _list_sample_images(stage_key)
    sample_choice = st.selectbox("Sample images", ["-- none --"] + samples, label_visibility="collapsed")
    uploaded = st.file_uploader("Or upload your own", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

    image_path = None
    if uploaded is not None:
        os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
        image_path = os.path.join(TEMP_UPLOADS_DIR, uploaded.name)
        with open(image_path, 'wb') as f:
            f.write(uploaded.getbuffer())
    elif sample_choice != "-- none --":
        stage_folder = os.path.join(SAMPLE_IMAGES_DIR, stage_key)
        base = stage_folder if os.path.isdir(stage_folder) else SAMPLE_IMAGES_DIR
        image_path = os.path.join(base, sample_choice)

    if image_path and os.path.exists(image_path):
        st.image(image_path, caption="Selected image", width='stretch')

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        run_clicked = st.button("▶ Run", type="primary", disabled=(image_path is None), width='stretch')
    with col2:
        stop_clicked = st.button("⏹ Stop", disabled=not st.session_state.running, width='stretch')
    with col3:
        reset_clicked = st.button("↻ Reset", width='stretch')

    return image_path, run_clicked, stop_clicked, reset_clicked


def render(debug: bool = False):
    inject_animation_css()
    _init_state()

    with st.sidebar:
        image_path, run_clicked, stop_clicked, reset_clicked = render_sidebar_controls(debug)

    if run_clicked:
        _reset_run()
        st.session_state.selected_image_path = image_path
        st.session_state.running = True
        st.session_state.run_started_at = time.time()
        st.rerun()
    if stop_clicked:
        st.session_state.stop_requested = True
        st.session_state.running = False
        st.rerun()
    if reset_clicked:
        _reset_run()
        st.rerun()

    st.title("🦷 Panoramic X-Ray Analysis")
    disclaimer_banner()
    pipeline_status(_done_steps(st.session_state.anim_stage), STAGE_LABELS,
                     current=STAGE_LABELS[min(st.session_state.anim_stage, len(STAGE_LABELS) - 1)]
                     if st.session_state.running or st.session_state.result else None)
    st.divider()

    # ---- run the real pipeline once ----
    if st.session_state.running and st.session_state.result is None and not st.session_state.stop_requested:
        progress_msgs = [
            "📖 Reading the X-ray...", "🔷 Locating quadrants...", "🦷 Detecting teeth...",
            "🔍 Checking for disease...", "🧬 Classifying disease type...", "📊 Assessing severity...",
        ]
        progress_bar = st.progress(0, text=progress_msgs[0])
        status_placeholder = st.empty()

        try:
            models = get_models()
            for i, msg in enumerate(progress_msgs):
                if st.session_state.stop_requested:
                    status_placeholder.warning("⏹ Processing stopped by user.")
                    break
                progress_bar.progress(int(((i + 1) / len(progress_msgs)) * 90), text=msg)
                time.sleep(0.2)

            if not st.session_state.stop_requested:
                result, warnings = run_pipeline(
                    image_path=st.session_state.selected_image_path,
                    models=models,
                    class_names=CLASS_NAMES,
                    device='cuda',
                    conf_threshold=DETECTION_CONF_THRESHOLD_DEFAULT,
                )
                progress_bar.progress(100, text="✅ Complete!")
                st.session_state.result = result
                st.session_state.warnings = warnings

                duration = time.time() - (st.session_state.run_started_at or time.time())
                try:
                    report_id = save_report(
                        result=result, warnings=warnings,
                        stage=st.session_state.selected_stage,
                        image_path=st.session_state.selected_image_path,
                        duration_seconds=duration,
                        models_used={"class_names": CLASS_NAMES},
                    )
                    st.session_state.last_report_id = report_id
                except Exception as e:
                    st.session_state.warnings.append(f"Report could not be saved: {e}")

                time.sleep(0.4)
                progress_bar.empty()

        except AppError as e:
            st.session_state.running = False
            show_error(e, debug=debug)
            progress_bar.empty()
        except Exception as e:
            st.session_state.running = False
            show_error(e, debug=debug)
            progress_bar.empty()
        finally:
            st.session_state.running = False

    if st.session_state.warnings:
        st.warning("⚠️ **Warnings during processing:**")
        for w in st.session_state.warnings:
            st.caption(f"• {w}")

    result = st.session_state.result

    if result is not None and not st.session_state.stop_requested:
        quad_names = list(result.get('quadrant_images', {}).keys())
        quad_name_map = {full: full.split('_')[-1] for full in quad_names}

        # ---- STAGE 0: Quadrants ----
        if st.session_state.anim_stage == 0 and quad_names:
            st.subheader("🔷 Step 1 - Finding Quadrants")
            boxes_so_far = []
            for i in range(min(st.session_state.anim_item + 1, len(quad_names))):
                qname = quad_names[i]
                x1, y1, x2, y2 = result['quadrant_boxes'][quad_name_map[qname]]
                boxes_so_far.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': qname})

            img = result['original_image']
            h, w = img.shape[:2]
            render_image_with_overlay(_image_to_base64(img), w, h, boxes_so_far, box_color="#00e676")
            if debug:
                st.caption(f"Debug: image {w}x{h}px · {len(boxes_so_far)}/{len(quad_names)} quadrants shown")

            if st.session_state.anim_item < len(quad_names) - 1:
                st.session_state.anim_item += 1
                time.sleep(0.5)
                st.rerun()
            else:
                time.sleep(0.6)
                st.session_state.anim_stage, st.session_state.anim_item = 1, 0
                st.rerun()

        # ---- STAGE 1: Teeth ----
        elif st.session_state.anim_stage == 1 and quad_names:
            st.subheader("🦷 Step 2 - Detecting & Numbering Teeth")
            current_quad_idx = min(st.session_state.anim_item, len(quad_names) - 1)
            qname = quad_names[current_quad_idx]
            teeth = result['teeth_per_quadrant'].get(qname, [])

            if teeth:
                st.caption(f"**Quadrant:** {qname}")
                qimg = result['quadrant_images'][qname]
                h, w = qimg.shape[:2]
                boxes = [{'x1': t['box'][0], 'y1': t['box'][1], 'x2': t['box'][2], 'y2': t['box'][3],
                          'label': f"#{t['class_name']}"} for t in teeth]
                render_image_with_overlay(_image_to_base64(qimg), w, h, boxes, box_color="#ffab00")
                if debug:
                    st.caption(f"Debug: {len(teeth)} teeth in {qname}, crop {w}x{h}px")

            if current_quad_idx < len(quad_names) - 1:
                st.session_state.anim_item += 1
                time.sleep(0.7)
                st.rerun()
            else:
                time.sleep(0.6)
                st.session_state.anim_stage, st.session_state.anim_item = 2, 0
                st.rerun()

        # ---- STAGE 2: Health ----
        elif st.session_state.anim_stage == 2:
            st.subheader("🔍 Step 3 - Health Status Check")
            all_teeth = result.get('all_teeth', [])
            n_show = min(st.session_state.anim_item + 1, len(all_teeth))

            if all_teeth:
                shown = all_teeth[:n_show]
                for t in shown:
                    t['image_base64'] = _image_to_base64(t['image'])
                healthy_count = sum(1 for t in shown if t['status'] == 'Healthy')
                disease_count = sum(1 for t in shown if t['status'] == 'Disease Found')
                render_live_counter(healthy_count, disease_count)
                render_tooth_grid(shown, status_key='status', highlight='disease')

                if n_show < len(all_teeth):
                    st.session_state.anim_item += 1
                    time.sleep(0.25)
                    st.rerun()
                else:
                    time.sleep(0.8)
                    st.session_state.anim_stage, st.session_state.anim_item = 3, 0
                    st.rerun()
            else:
                st.info("No teeth data available")
                st.session_state.anim_stage = 5
                st.rerun()

        # ---- STAGE 3: Disease ----
        elif st.session_state.anim_stage == 3:
            st.subheader("🧬 Step 4 - Disease Classification")
            diseased = result.get('diseased_teeth', [])
            if not diseased:
                st.success("✅ No diseased teeth detected!")
                time.sleep(0.5)
                st.session_state.anim_stage = 5
                st.rerun()
            else:
                n_show = min(st.session_state.anim_item + 1, len(diseased))
                shown = diseased[:n_show]
                for t in shown:
                    t['image_base64'] = _image_to_base64(t['image'])
                render_tooth_grid(shown, status_key='disease', highlight='')

                if n_show < len(diseased):
                    st.session_state.anim_item += 1
                    time.sleep(0.35)
                    st.rerun()
                else:
                    time.sleep(0.6)
                    st.session_state.anim_stage, st.session_state.anim_item = 4, 0
                    st.rerun()

        # ---- STAGE 4: Severity ----
        elif st.session_state.anim_stage == 4:
            caries_teeth = [t for t in result.get('diseased_teeth', []) if t.get('caries_severity')]
            if caries_teeth:
                st.subheader("📊 Step 5 - Caries Severity Assessment")
                n_show = min(st.session_state.anim_item + 1, len(caries_teeth))
                shown = caries_teeth[:n_show]
                for t in shown:
                    t['image_base64'] = _image_to_base64(t['image'])
                render_tooth_grid(shown, status_key='caries_severity', highlight='')

                if n_show < len(caries_teeth):
                    st.session_state.anim_item += 1
                    time.sleep(0.35)
                    st.rerun()
                else:
                    time.sleep(0.6)
                    st.session_state.anim_stage = 5
                    st.rerun()
            else:
                st.session_state.anim_stage = 5
                st.rerun()

        # ---- STAGE 5: Final Summary ----
        else:
            st.subheader("✅ Final Summary")
            diseased = result.get('diseased_teeth', [])
            all_teeth = result.get('all_teeth', [])
            metric_row([
                ("🦷 Total Teeth", len(all_teeth)),
                ("✅ Healthy", len(all_teeth) - len(diseased)),
                ("⚠️ Diseased", len(diseased)),
            ])

            if all_teeth:
                boxes = []
                for t in diseased:
                    qkey = t['quad_key'].split('_')[-1]
                    qx1, qy1, _, _ = result['quadrant_boxes'][qkey]
                    tx1, ty1, tx2, ty2 = t['box']
                    final_label = t.get('caries_severity') or t.get('disease', 'Unknown')
                    boxes.append({'x1': qx1 + tx1, 'y1': qy1 + ty1, 'x2': qx1 + tx2, 'y2': qy1 + ty2,
                                  'label': f"#{t['class_name']} - {final_label}"})
                img = result['original_image']
                h, w = img.shape[:2]
                render_image_with_overlay(_image_to_base64(img), w, h, boxes, box_color="#f44336")

            if diseased:
                st.subheader("📋 Details")
                df = pd.DataFrame([{
                    'Quadrant': t['quad_key'], 'Tooth #': t['class_name'],
                    'Disease': t.get('caries_severity') or t['disease'],
                    'Confidence': f"{max(t.get('disease_probs', {}).values()):.0%}" if t.get('disease_probs') else " - ",
                } for t in diseased])
                st.dataframe(df, width='stretch')

            if st.session_state.last_report_id:
                st.success(f"📁 Saved to Reports / History as `{st.session_state.last_report_id}`")

    elif st.session_state.stop_requested:
        st.info("⏹ Processing stopped. Choose another image to continue.")
        if st.button("Reset & Start Over"):
            _reset_run()
            st.rerun()

    else:
        st.markdown("""
            ### 👋 Welcome!
            Select an image from the sidebar to begin analysis.

            **How it works:**
            1. Choose which model stage to run
            2. Upload or select a sample image
            3. Watch the animated analysis unfold
            4. Review results - automatically saved to **Reports / History**
        """)
