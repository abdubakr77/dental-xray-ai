"""
Dental X-ray AI - main entry point.

This file only does navigation and page config. Every model call, every
piece of pipeline logic, lives in the original inference_pipeline.py /
model_utils.py / utils.py / vis.py / animation.py (untouched, except one
CSS bug fix in animation.py - see its top-of-function comment) plus the
thin integration layer under core/ and storage/.
"""

import streamlit as st

from core.config import APP_TITLE, NAV_PAGES
from ui import dashboard, analysis, models_playground, reports_page, about

st.set_page_config(
    page_title="Dental X-ray AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Analysis"
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

with st.sidebar:
    st.markdown(f"## 🦷 {APP_TITLE.split(' - ')[0].strip()}")
    st.caption(APP_TITLE.split(' - ')[-1].strip())
    st.session_state.nav_page = st.radio("Navigate", NAV_PAGES,
                                          index=NAV_PAGES.index(st.session_state.nav_page))
    st.divider()
    st.session_state.debug_mode = st.toggle("🛠 Debug mode", value=st.session_state.debug_mode,
                                             help="Show raw coordinates, tensor/class info, and full tracebacks.")
    st.divider()

debug = st.session_state.debug_mode

PAGES = {
    "Dashboard": dashboard.render,
    "Analysis": analysis.render,
    "Individual Models": models_playground.render,
    "Reports / History": reports_page.render,
    "About": about.render,
}

PAGES[st.session_state.nav_page](debug=debug)
