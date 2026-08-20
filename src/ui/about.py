"""
About / System Info page.

The model performance numbers here are copied verbatim from the existing
app.py 'Reports' tab (they were hardcoded there too - this app has no
mechanism to compute them itself, since they come from held-out test-set
evaluation done at training time, not from anything the runtime pipeline
produces). Confusion matrix images are read from the same paths as before.
"""

import os
import streamlit as st
import pandas as pd
from src.core.model_registry import device_info
from src.core.config import CONFIG_PATH,_PROJECT_ROOT 


def render(debug: bool = False):
    st.title("ℹ️ About / System Info")

    st.markdown("""
    **Dental AI - Panoramic X-Ray Intelligence System**

    A five-stage pipeline: quadrant detection → tooth detection/enumeration →
    healthy/unhealthy classification → disease classification → caries severity
    classification. Detection stages use YOLO; classification stages use a
    Swin Transformer (SwinV2-T) backbone.
    """)

    st.divider()
    st.subheader("🎯 Detection Models (YOLO)")
    yolo_df = pd.DataFrame({
        'mAP50': [0.99500, 0.92942, 0.97856],
        'mAP50-95': [0.72087, 0.56098, 0.78198],
        'Precision': [0.99879, 0.93218, 0.97093],
        'Recall': [0.99568, 0.92044, 0.95918],
    }, index=['Quadrant Detection', 'Tooth Enumeration', 'Teeth Refinement'])
    st.dataframe(yolo_df, width='stretch')

    st.subheader("🔬 Classification Models")
    clf_df = pd.DataFrame({
        'Accuracy': [0.7143, 0.8858, 0.8273],
        'Precision (avg)': [0.7096, 0.7270, 0.9062],
        'Recall (avg)': [0.7315, 0.8682, 0.8744],
        'F1-Score (avg)': [0.7204, 0.7735, 0.8900],
    }, index=['Healthy/Unhealthy', 'Disease Type', 'Caries Severity'])
    st.dataframe(clf_df, width='stretch')

    st.divider()
    st.subheader("🔍 Confusion Matrices")

    cm_paths = {
        'Healthy/Unhealthy': '../Runs/Stage 3/Healthy & Un-Healthy Classifier/confusion_matrix.png',
        'Disease Type': '../Runs/Stage 3/Disease Classifier/confusion_matrix.png',
        'Caries Severity': '../Runs/Stage 3/Caries & Deep Caries Classifier/confusion_matrix.png',
    }

    cols = st.columns(3)
    for col, (name, path) in zip(cols, cm_paths.items()):
        with col:
            st.caption(name)
            
            if os.path.exists(path):
                st.image(path, width='stretch')
            else:
                fixed_path = os.path.join(_PROJECT_ROOT, path.replace("../", "", 1))
                
                if os.path.exists(fixed_path):
                    st.image(fixed_path, width='stretch')
                else:
                    st.warning("📁 File not found")

    st.divider()
    st.subheader("⚠️ Model Limitations")
    with st.expander("Read more", expanded=False):
        st.markdown("""
        **Rare Classes (Periapical):** Only 16 test samples. Precision: 37.5%,
        Recall: 75%. Use as a screening signal, not definitive diagnosis.

        **Severity Boundary:** Deep caries detection (Recall: 64%, Precision: 56%)
        is weaker than caries detection. The boundary between severity levels
        is continuous, not discrete.

        **Pipeline Error Propagation:** Early-stage errors cascade forward and
        cannot be corrected. End-to-end accuracy is lower than individual stages.

        **Classification Bottleneck:** Healthy/Unhealthy stage at 71.4% accuracy
        is the highest-traffic step. Improving this is the clearest path to
        better pipeline reliability.

        **Clinical Disclaimer:** This is a **screening aid only**, not a diagnostic
        tool. All positive findings must be confirmed by a qualified dentist.
        """)

    if debug:
        st.divider()
        st.subheader("🛠 Debug: System Info")
        st.json(device_info())
        st.caption(f"Model config path: {CONFIG_PATH}")

    st.divider()
    f1, f2, f3 = st.columns(3)
    f1.markdown("[📦 GitHub](https://github.com/abdubakr77/dental-xray-ai)")
    f2.markdown("[💼 LinkedIn](https://linkedin.com/in/abdubakr) | [🌐 Portfolio](https://abdubakr77.github.io)")
    f3.markdown("**Built by Abdullah Bakr** | 2026-2027")
