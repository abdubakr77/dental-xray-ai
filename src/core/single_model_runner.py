"""
Run exactly one model in isolation, for the Individual Models playground.

Nothing here reimplements detection or classification. Every call below
delegates to the same functions inference_pipeline.run_pipeline() already
uses for the full pipeline - this module only wires a single stage's input
and output without running the rest of the chain.
"""

import os
import time
import tempfile
import shutil

import cv2
import pandas as pd
from torch import utils as torch_utils

from model_utils import export_quadrants_using_quad_model, export_teeth_in_quad_using_enum_model
from inference_pipeline import (
    crop_teeth_from_merged_label, SingleCropDataset, CLASSIFIER_TRANSFORM,
)
from model_utils import predict_classifier
from core.config import QUADRANT_NAMES, DETECTION_CONF_THRESHOLD_DEFAULT
from core.model_registry import device_info


def _device_str() -> str:
    info = device_info()
    return "cuda" if info["cuda_available"] else "cpu"


def run_quadrant_only(image_path: str, models: dict, conf_threshold: float = DETECTION_CONF_THRESHOLD_DEFAULT) -> dict:
    """Quadrant detector on one full panoramic image. Returns boxes + crops + log + timing."""
    session_dir = tempfile.mkdtemp()
    try:
        img_folder = os.path.join(session_dir, 'input')
        os.makedirs(img_folder)
        img_name = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(img_folder, img_name))

        quad_out = os.path.join(session_dir, 'quad_out')
        dummy_df = pd.DataFrame({'File_Name': [img_name]})

        t0 = time.time()
        log_df = export_quadrants_using_quad_model(
            models['quadrant_model'], img_folder, annotations_df=dummy_df,
            output_root=quad_out, conf_threshold=conf_threshold,
            clear_existing=False, export_labels=False, export_images=True, verbose=False)
        elapsed = time.time() - t0

        quadrant_boxes = {
            row['quad'].replace(' ', ''): (row['x1'], row['y1'], row['x2'], row['y2'])
            for row in log_df.to_dict('records') if row['event'] == 'successful_detection'
        }
        quadrant_images = {}
        images_dir = os.path.join(quad_out, 'images')
        if os.path.isdir(images_dir):
            for fname in os.listdir(images_dir):
                quadrant_images[os.path.splitext(fname)[0]] = cv2.cvtColor(
                    cv2.imread(os.path.join(images_dir, fname)), cv2.COLOR_BGR2RGB)

        original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return {
            'original_image': original,
            'quadrant_boxes': quadrant_boxes,
            'quadrant_images': quadrant_images,
            'log': log_df.to_dict('records'),
            'inference_seconds': elapsed,
        }
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


def run_teeth_only(crop_image_path: str, quadrant_name: str, models: dict,
                    conf_threshold: float = DETECTION_CONF_THRESHOLD_DEFAULT) -> dict:
    """
    Tooth/enumeration detector on a single quadrant crop. `quadrant_name` must
    be one of core.config.QUADRANT_NAMES - the enum model recovers which
    quadrant a crop is from by filename suffix (see export_teeth_in_quad_using_enum_model),
    so the crop is renamed to match that convention before running.
    """
    if quadrant_name not in QUADRANT_NAMES:
        raise ValueError(f"quadrant_name must be one of {QUADRANT_NAMES}")

    session_dir = tempfile.mkdtemp()
    try:
        images_root = os.path.join(session_dir, 'images')
        labels_root = os.path.join(session_dir, 'labels')  # left empty: no GT, pure inference
        os.makedirs(images_root)

        tag = quadrant_name.replace(' ', '')
        dest_name = f"crop_{tag}.png"
        shutil.copy(crop_image_path, os.path.join(images_root, dest_name))

        teeth_out = os.path.join(session_dir, 'teeth_out')

        t0 = time.time()
        log_df = export_teeth_in_quad_using_enum_model(
            models['enumeration_model'], images_root=images_root, labels_root=labels_root,
            output_root=teeth_out, conf_threshold=conf_threshold,
            background_area_ratio_high=5, leak_area_ratio=0.3,
            clear_existing=False, export_labels=True, export_images=True, verbose=False)
        elapsed = time.time() - t0

        base_name = os.path.splitext(dest_name)[0]
        teeth = crop_teeth_from_merged_label(
            os.path.join(teeth_out, 'images', dest_name),
            os.path.join(teeth_out, 'labels', base_name + '.txt'),
            [str(i) for i in range(8)]
        )
        crop_img = cv2.cvtColor(cv2.imread(crop_image_path), cv2.COLOR_BGR2RGB)
        return {
            'quadrant_image': crop_img,
            'teeth': teeth,
            'log': log_df.to_dict('records') if log_df is not None else [],
            'inference_seconds': elapsed,
        }
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)
