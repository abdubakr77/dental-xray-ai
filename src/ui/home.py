"""
Home page - the landing experience (was previously a near-empty center
column with a wall of "how it works" text; see screenshot 1). Real content
now lives here: what the app does, the pipeline at a glance, and a way to
jump straight into an analysis. Detailed instructions/FAQ live on the
dedicated Help page instead of crowding this one.
"""

import os
import random
import streamlit as st

from core.config import SAMPLE_IMAGES_DIR, DISCLAIMER
from animation import inject_animation_css

PIPELINE_STEPS = [
    ("🩻", "Panoramic X-Ray", "The full mouth image you upload"),
    ("🔷", "Quadrant Detection", "4 regions: upper/lower, left/right"),
    ("🦷", "Tooth Detection", "Every individual tooth, numbered 0-7 per quadrant"),
    ("🔍", "Health Classification", "Healthy or requires further review"),
    ("🧬", "Disease Classification", "Impacted, Caries, or Periapical"),
    ("📊", "Severity Classification", "Caries teeth only: Caries or Deep Caries"),
]


def _list_sample_images() -> list:
    """A random pick (up to 4), chosen once per session - not re-rolled on
    every rerun. Re-rolling every render was the actual bug: clicking a
    sample's own button also triggers a rerun, so a fresh random pick would
    replace the set on that very click, before the app even navigated away."""
    if not os.path.isdir(SAMPLE_IMAGES_DIR):
        return []
    if 'home_sample_pick' not in st.session_state:
        all_samples = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.session_state['home_sample_pick'] = random.sample(all_samples, k=min(4, len(all_samples)))
    return st.session_state['home_sample_pick']


def _go_to_analysis(sample_filename: str = None):
    if sample_filename:
        st.session_state['pending_sample_image'] = sample_filename
    st.session_state['requested_nav_page'] = 'Analysis'


def render(debug: bool = False):
    inject_animation_css()

    st.markdown("""
        <div style="text-align:center; padding: 18px 0 8px 0;">
            <div style="font-size:2.6rem;">🦷</div>
            <div style="font-size:2.1rem; font-weight:800; margin-top:4px;">Dental AI</div>
            <div style="font-size:1.05rem; opacity:0.75; margin-top:4px;">
                Panoramic X-Ray Intelligence System
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='max-width:720px; margin:0 auto; text-align:center; opacity:0.85; font-size:0.95rem;'>"
        "A five-stage computer-vision pipeline that locates dental quadrants, detects and numbers "
        "individual teeth, and classifies each one for health status, disease type, and (for caries) "
        "severity - visualized stage by stage as it runs."
        "</div>",
        unsafe_allow_html=True
    )
    st.caption(f"<div style='text-align:center;'>ℹ️ {DISCLAIMER}</div>", unsafe_allow_html=True)

    st.write("")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        if st.button("▶  Start an Analysis", type="primary", width='stretch'):
            _go_to_analysis()
            st.rerun()

    st.divider()
    st.markdown("#### How the pipeline works")
    cols = st.columns(len(PIPELINE_STEPS))
    for col, (icon, title, desc) in zip(cols, PIPELINE_STEPS):
        with col:
            st.markdown(
                f"<div style='text-align:center; padding:10px 6px; border-radius:12px; "
                f"background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); "
                f"min-height:132px;'>"
                f"<div style='font-size:1.6rem;'>{icon}</div>"
                f"<div style='font-weight:700; font-size:0.82rem; margin-top:6px;'>{title}</div>"
                f"<div style='font-size:0.72rem; opacity:0.65; margin-top:4px;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    samples = _list_sample_images()
    if samples:
        st.divider()
        st.markdown("#### Try a sample image")
        st.caption("Pick one to jump straight into analysis with it pre-selected.")
        cols = st.columns(len(samples))
        for i, (col, fname) in enumerate(zip(cols, samples)):
            with col:
                with st.container(key=f"home_sample_card_{i}"):
                    st.image(os.path.join(SAMPLE_IMAGES_DIR, fname), width='stretch')
                if st.button("Analyze this", key=f"home_sample_{fname}", width='stretch'):
                    _go_to_analysis(fname)
                    st.rerun()

    st.divider()
    h1, h2 = st.columns([3, 1])
    with h1:
        st.caption("New here? The **Help** page covers input requirements, what each result means, "
                   "and troubleshooting in detail.")
    with h2:
        if st.button("Open Help →", width='stretch'):
            st.session_state['requested_nav_page'] = 'Help'
            st.rerun()
