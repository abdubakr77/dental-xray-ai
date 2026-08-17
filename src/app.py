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
from ui import dashboard, analysis, models_playground, reports_page, about, home, help as help_page

st.set_page_config(
    page_title="Dental X-ray AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Other pages request a page switch via 'requested_nav_page' instead of
# writing 'nav_page' directly - Streamlit forbids writing to a key after its
# widget has been instantiated in the same run, and by the time a page's
# render() runs (below), the nav_page radio has already been created this
# run. Draining the request here, before that widget exists, is the one
# point where reassigning nav_page is actually allowed.
if "requested_nav_page" in st.session_state:
    st.session_state.nav_page = st.session_state.pop("requested_nav_page")

with st.sidebar:
    st.markdown(f"## 🦷 {APP_TITLE.split(' - ')[0].strip()}")
    st.caption(APP_TITLE.split(' - ')[-1].strip())
    st.radio("Navigate", NAV_PAGES, key="nav_page")
    st.divider()
    st.toggle("🛠 Debug mode", key="debug_mode",
              help="Show raw coordinates, tensor/class info, and full tracebacks.")
    st.divider()

debug = st.session_state.debug_mode

PAGES = {
    "Home": home.render,
    "Dashboard": dashboard.render,
    "Analysis": analysis.render,
    "Individual Models": models_playground.render,
    "Reports / History": reports_page.render,
    "Help": help_page.render,
    "About": about.render,
}

# Wrapping the page in a keyed container gives it a stable "st-key-page_fade_<page>"
# CSS class (see animation.py) that we can target with a soft fade+rise
# entrance - the closest approximation of a page transition Streamlit's
# rerun-the-whole-script model allows, since there's no persisting DOM to
# animate an "exit" for. The key includes the page name on purpose: a fixed
# key would keep reusing the same DOM node across navigations, and a CSS
# `animation` only replays when its element is actually (re)inserted into
# the DOM - so a fixed key meant the fade played once, ever, then every
# later switch looked instant. Best-effort: if a future Streamlit release
# changes that class naming, this just silently stops animating.
with st.container(key=f"page_fade_{st.session_state.nav_page.replace(' ', '_').replace('/', '_')}"):
    PAGES[st.session_state.nav_page](debug=debug)
