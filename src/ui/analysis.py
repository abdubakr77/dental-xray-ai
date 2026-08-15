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
