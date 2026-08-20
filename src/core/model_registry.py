"""
Model loading, cached once per Streamlit session/process.

This module does not reimplement model loading. It calls the existing
`load_recommended_models` from inference_pipeline.py exactly as app.py did,
and wraps it with:
  - st.cache_resource so models load once, not on every rerun
  - a clear, user-facing error if the config/model files are missing
  - a small metadata readout (device, model paths) for the About / Debug views
"""

import os
import yaml
import streamlit as st
import torch

from src.inference_pipeline import load_recommended_models
from src.core.config import CONFIG_PATH, DEVICE_PREFERENCE
from src.core.errors import AppError


@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_models():
    """
    Load and cache all pipeline models. Raises AppError with a clear message
    on any failure instead of letting a raw traceback reach the UI.
    """
    if not os.path.exists(CONFIG_PATH):
        raise AppError(
            f"Model config not found at `{CONFIG_PATH}`. "
            "Set DENTAL_AI_MODEL_CONFIG or place trained_models.yaml at that path.",
            technical=f"CONFIG_PATH={CONFIG_PATH} does not exist"
        )

    try:
        with open(CONFIG_PATH, 'r') as f:
            all_models = yaml.safe_load(f)
    except Exception as e:
        raise AppError(f"Could not parse model config `{CONFIG_PATH}`.", technical=str(e))

    if 'final_recommended_models' not in all_models:
        raise AppError(
            f"`{CONFIG_PATH}` is missing the `final_recommended_models` key.",
            technical=f"Keys found: {list(all_models.keys())}"
        )

    try:
        return load_recommended_models(
            all_models['final_recommended_models'],
            device=DEVICE_PREFERENCE
        )
    except FileNotFoundError as e:
        raise AppError(
            "One or more model files listed in the config could not be found on disk.",
            technical=str(e)
        )
    except Exception as e:
        raise AppError("Model loading failed.", technical=str(e))


def device_info() -> dict:
    """Small readout used in the About page and Debug mode."""
    cuda_available = torch.cuda.is_available()
    return {
        "requested_device": DEVICE_PREFERENCE,
        "resolved_device": DEVICE_PREFERENCE if cuda_available else "cpu (CUDA unavailable, fell back)",
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "config_path": CONFIG_PATH,
        "config_exists": os.path.exists(CONFIG_PATH),
    }
