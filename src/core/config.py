"""
Centralized configuration for the Dental AI app.

Everything here is either:
  (a) copied verbatim from the existing app.py (CLASS_NAMES, STAGE_* dicts,
      CONFIG_PATH), or
  (b) new app-level settings (report storage location, app title) that don't
      touch model behavior at all.

Nothing about model architecture, preprocessing, class ordering, or
coordinate systems is redefined here - this module only centralizes where
those existing values live so the rest of the app doesn't hardcode them
in five different places.
"""

import os

# ---------------------------------------------------------------------------
# Path anchoring.
#
# Every path below used to be a plain relative string ("sample_images",
# "../configs/trained_models.yaml", ...). Relative paths in Python resolve
# against the process's CURRENT WORKING DIRECTORY at the moment it was
# launched - NOT against wherever app.py itself lives on disk. That only
# happens to work if you `cd` into the exact folder the author had in mind
# before running `streamlit run app.py`; run it from anywhere else (e.g.
# from the project root instead of from src/) and paths like
# SAMPLE_IMAGES_DIR silently resolve to a folder that doesn't exist, so
# nothing shows up and no error is raised either.
#
# Anchoring to this file's own location makes every path below correct
# regardless of which directory `streamlit run` was invoked from.
#
# Layout assumed (matches the project structure this app ships in):
#   <PROJECT_ROOT>/
#       configs/trained_models.yaml
#       src/                    <- _SRC_DIR (this file's parent's parent)
#           core/config.py      <- this file
#           sample_images/
#           reports/
#           temp_uploads/
#           app.py
# If your own layout differs, override with the env vars below rather than
# editing these constants.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# ---------------------------------------------------------------------------
# Model config - same path convention as the existing app.py, just anchored
# absolutely instead of left relative to an assumed working directory.
# The YAML itself (paths under `final_recommended_models`) was not provided,
# so its contents are not fabricated here; this only points at where it lives.
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get(
    "DENTAL_AI_MODEL_CONFIG",
    os.path.join(_PROJECT_ROOT, "configs", "trained_models.yaml")
)
DEVICE_PREFERENCE = "cuda"  # existing load_recommended_models() falls back to cpu internally

SAMPLE_IMAGES_DIR = os.path.join(_SRC_DIR, "sample_images")

# NOTE: the model paths INSIDE trained_models.yaml (quadrant_model: ../Runs/...)
# are still read and used as-is by the existing load_recommended_models() /
# YOLO() / torch.load() calls in inference_pipeline.py and model_utils.py,
# which were not modified - those stay relative to the CURRENT WORKING
# DIRECTORY the app was launched from (that part of the pipeline is
# untouched, by design). For those to resolve, run streamlit from inside
# src/ (`cd src && streamlit run app.py`), matching the working directory
# the YAML's own relative paths were written against.

# ---------------------------------------------------------------------------
# Class names - copied unchanged from the existing app.py. These must stay in
# sync with the class ordering the models were trained/exported with.
# Do NOT reorder, rename, or remap. If your model config changes these,
# change them here once, not in the UI code.
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    'teeth': [str(i) for i in range(8)],
    'healthy_unhealthy': ['Disease Found', 'Healthy'],
    'disease': ['Impacted', 'Caries', 'Periapical'],
    'caries_severity': ['Caries', 'Deep Caries'],
}

QUADRANT_NAMES = ["Upper Right", "Upper Left", "Lower Left", "Lower Right"]

# ---------------------------------------------------------------------------
# Pipeline stages - copied unchanged from the existing app.py.
# ---------------------------------------------------------------------------
STAGE_OPTIONS = {
    'Full Pipeline': 'full',
    'Quadrant Detection Only': 'quadrant',
    'Teeth Detection Only': 'teeth',
    'Healthy / Unhealthy Only': 'healthy_unhealthy',
    'Disease Type Only': 'disease',
    'Caries Severity Only': 'caries_severity',
}

STAGE_REQUIRED_INPUT = {
    'full': "full panoramic X-ray",
    'quadrant': "full panoramic X-ray",
    'teeth': "single quadrant crop",
    'healthy_unhealthy': "single tooth image",
    'disease': "single tooth image (diseased)",
    'caries_severity': "single tooth image (with caries)",
}

# Human-readable description of what each individual model needs, shown in
# the Individual Models playground so a user can't unknowingly feed the
# wrong input into a model that expects something else.
STAGE_INPUT_HELP = {
    'quadrant': "A full panoramic X-ray. The quadrant detector expects the whole mouth, not a crop.",
    'teeth': "A single quadrant crop (e.g. 'Upper Right'), already isolated from the full panoramic image.",
    'healthy_unhealthy': "A single cropped tooth image, isolated from its quadrant.",
    'disease': "A single cropped tooth image that is already known/assumed to be unhealthy.",
    'caries_severity': "A single cropped tooth image that is already classified as Caries.",
}

STAGE_SEQUENCE = {
    'full': ['quadrant', 'teeth', 'healthy_unhealthy', 'disease', 'caries_severity'],
    'quadrant': ['teeth', 'healthy_unhealthy', 'disease', 'caries_severity'],
    'teeth': ['healthy_unhealthy', 'disease', 'caries_severity'],
    'healthy_unhealthy': ['disease', 'caries_severity'],
    'disease': ['caries_severity'],
    'caries_severity': [],
}

DETECTION_CONF_THRESHOLD_DEFAULT = 0.3

# ---------------------------------------------------------------------------
# App-level settings (new - not part of the ML pipeline).
# ---------------------------------------------------------------------------
APP_TITLE = "Dental AI - Panoramic X-Ray Intelligence System"
REPORTS_DIR = os.environ.get("DENTAL_AI_REPORTS_DIR", os.path.join(_SRC_DIR, "reports"))
TEMP_UPLOADS_DIR = os.path.join(_SRC_DIR, "temp_uploads")

DISCLAIMER = (
    "AI-generated results are for research / decision-support purposes only "
    "and are not a substitute for professional dental diagnosis. All findings "
    "must be confirmed by a qualified dentist."
)

NAV_PAGES = ["Home", "Dashboard", "Analysis", "Individual Models", "Reports / History", "Help", "About"]
