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
# Model config - same path convention as the existing app.py.
# The YAML itself (paths under `final_recommended_models`) was not provided,
# so it is not fabricated here; this only points at where it lives.
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("DENTAL_AI_MODEL_CONFIG", "../configs/trained_models.yaml")
DEVICE_PREFERENCE = "cuda"  # existing load_recommended_models() falls back to cpu internally

SAMPLE_IMAGES_DIR = "../sample_images"

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
REPORTS_DIR = os.environ.get("DENTAL_AI_REPORTS_DIR", "reports")
TEMP_UPLOADS_DIR = "temp_uploads"

DISCLAIMER = (
    "AI-generated results are for research / decision-support purposes only "
    "and are not a substitute for professional dental diagnosis. All findings "
    "must be confirmed by a qualified dentist."
)

NAV_PAGES = ["Dashboard", "Analysis", "Individual Models", "Reports / History", "About"]
