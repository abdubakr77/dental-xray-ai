"""
Individual Models playground.

Lets a user run exactly one model in isolation, with explicit input
validation so a full panoramic image can't accidentally be fed to a
classifier expecting a single tooth crop, etc.
"""

import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from core.config import CLASS_NAMES, QUADRANT_NAMES, STAGE_INPUT_HELP, DETECTION_CONF_THRESHOLD_DEFAULT
from core.model_registry import get_models
from core.errors import AppError, show_error, validate_image_file
from core.single_model_runner import (
    run_quadrant_only, run_teeth_only, run_healthy_unhealthy_only,
    run_disease_only, run_caries_severity_only,
)
from ui.components import probability_bars, metric_row, disclaimer_banner
from animation import inject_animation_css, render_image_with_overlay

MODEL_OPTIONS = {
    "Quadrant Detection": "quadrant",
    "Tooth Detection (Enumeration)": "teeth",
    "Healthy / Unhealthy Classification": "healthy_unhealthy",
    "Disease Classification": "disease",
    "Caries / Deep Caries Classification": "caries_severity",
}


def _load_uploaded_image(uploaded_file) -> np.ndarray:
    validate_image_file(uploaded_file)
    pil_img = Image.open(uploaded_file).convert("RGB")
    return np.array(pil_img)


def render(debug: bool = False):
    inject_animation_css()
    st.title("🧪 Individual Model Playground")
    st.caption("Run any single model in isolation - useful for debugging and for demoing one stage at a time.")
    disclaimer_banner()
    st.divider()

    label = st.selectbox("Select a model to test", list(MODEL_OPTIONS.keys()))
    model_key = MODEL_OPTIONS[label]

    st.info(f"**Required input:** {STAGE_INPUT_HELP[model_key]}")

    quadrant_choice = None
    if model_key == "teeth":
        quadrant_choice = st.selectbox("Which quadrant is this crop from?", QUADRANT_NAMES)

    uploaded = st.file_uploader("Upload the required input image", type=['png', 'jpg', 'jpeg'])
    if uploaded is None:
        st.stop()

    try:
        image = _load_uploaded_image(uploaded)
    except AppError as e:
        show_error(e, debug=debug)
        st.stop()

    st.image(image, caption="Input image", width='stretch')

    if not st.button("▶ Run this model", type="primary"):
        return

    try:
        models = get_models()
    except AppError as e:
        show_error(e, debug=debug)
        return

    tmp_path = os.path.join("temp_uploads", f"playground_{uploaded.name}")
    os.makedirs("temp_uploads", exist_ok=True)
    cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    st.divider()
    try:
        with st.spinner(f"Running {label}..."):

            if model_key == "quadrant":
                out = run_quadrant_only(tmp_path, models, DETECTION_CONF_THRESHOLD_DEFAULT)
                st.subheader("Result")
                metric_row([
                    ("Quadrants found", len(out['quadrant_boxes'])),
                    ("Inference time", f"{out['inference_seconds']:.2f}s"),
                ])
                h, w = out['original_image'].shape[:2]
                boxes = [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': name}
                         for name, (x1, y1, x2, y2) in out['quadrant_boxes'].items()]
                render_image_with_overlay(
                    _b64(out['original_image']), w, h, boxes, box_color="#00e676"
                )
                if debug:
                    st.dataframe(out['log'], width='stretch')

            elif model_key == "teeth":
                out = run_teeth_only(tmp_path, quadrant_choice, models, DETECTION_CONF_THRESHOLD_DEFAULT)
                st.subheader("Result")
                metric_row([
                    ("Teeth found", len(out['teeth'])),
                    ("Inference time", f"{out['inference_seconds']:.2f}s"),
                ])
                h, w = out['quadrant_image'].shape[:2]
                boxes = [{'x1': t['box'][0], 'y1': t['box'][1], 'x2': t['box'][2], 'y2': t['box'][3],
                          'label': f"#{t['class_name']}"} for t in out['teeth']]
                render_image_with_overlay(_b64(out['quadrant_image']), w, h, boxes, box_color="#ffab00")
                if debug:
                    st.dataframe(out['log'], width='stretch')

            elif model_key == "healthy_unhealthy":
                out = run_healthy_unhealthy_only(image, models, CLASS_NAMES['healthy_unhealthy'])
                _show_classifier_result(label, models['teeth_status_model'], out)

            elif model_key == "disease":
                out = run_disease_only(image, models, CLASS_NAMES['disease'])
                _show_classifier_result(label, models['disease_model'], out)

            elif model_key == "caries_severity":
                out = run_caries_severity_only(image, models, CLASS_NAMES['caries_severity'])
                _show_classifier_result(label, models['caries_status_model'], out)

    except AppError as e:
        show_error(e, debug=debug)
    except Exception as e:
        show_error(e, debug=debug)


def _b64(img_array) -> str:
    import io, base64
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _show_classifier_result(model_label: str, model_obj, out: dict):
    st.subheader("Result")
    top_class = out['prediction']
    top_prob = out['probabilities'][top_class]
    metric_row([
        ("Top prediction", top_class),
        ("Confidence", f"{top_prob*100:.1f}%"),
        ("Inference time", f"{out['inference_seconds']:.3f}s"),
    ])
    probability_bars(out['probabilities'], title="Full probability distribution")
    with st.expander("Model metadata"):
        st.write({"model_class": type(model_obj).__name__, "model_label": model_label})
