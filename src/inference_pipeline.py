import cv2
import os
import sys
# add the project root to the path so the src package can be imported
sys.path.append(os.path.abspath('..'))
from torch import utils, float32
import tempfile, shutil
from src.model_utils import predict_classifier, export_quadrants_using_quad_model, export_teeth_in_quad_using_enum_model
from torchvision.transforms import v2
import pandas as pd

class SingleCropDataset(utils.data.Dataset):
    """Wraps in-memory crops as a minimal Dataset so predict_classifier can run
    on live, unlabeled inference crops the same way it runs on a real test set."""
    def __init__(self, images, transform, classes):
        self.images = images
        self.transform = transform
        self.classes = classes  # predict_classifier's show_plot path reads dataset.classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), 0  # dummy label,


def crop_teeth_from_merged_label(image_path, label_path, class_names):
    """
    Reads a quadrant crop and its merged label file (from
    export_teeth_in_quad_using_enum_model) and crops out each individual tooth.

    Returns:
        list of {'class_name': str, 'box': [x1,y1,x2,y2] in quadrant-crop pixels, 'image': ndarray}
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    teeth = []
    if not os.path.exists(label_path):
        return teeth

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            cx, cy, nw, nh = map(float, parts[1:5])
            x1, y1 = int((cx - nw / 2) * w), int((cy - nh / 2) * h)
            x2, y2 = int((cx + nw / 2) * w), int((cy + nh / 2) * h)
            teeth.append({'class_name': class_names[cls_id], 'box': [x1, y1, x2, y2],
                           'image': img[max(0, y1):y2, max(0, x1):x2]})
    return teeth


CLASSIFIER_TRANSFORM = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_pipeline(image_path, models, class_names, device='cuda', conf_threshold=0.3):
    """
    Runs the full pipeline on one uploaded image, using export_quadrants_using_quad_model
    and export_teeth_in_quad_using_enum_model for detection (with all their built-in
    filtering/logging), and predict_classifier for every classifier stage.
    """
    session_dir = tempfile.mkdtemp()
    warnings = []
    result = {'quadrants': {}, 'teeth_per_quadrant': {}, 'diseased_teeth': []}

    try:
        # ---- Stage 1: quadrant detection ----
        img_folder = os.path.join(session_dir, 'input')
        os.makedirs(img_folder)
        img_name = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(img_folder, img_name))

        quad_out = os.path.join(session_dir, 'quad_out')
        dummy_df = pd.DataFrame({'File_Name': [img_name]})

        quad_log_df = export_quadrants_using_quad_model(
            models['quadrant_model'], img_folder, annotations_df=dummy_df,
            output_root=quad_out, conf_threshold=conf_threshold,
            clear_existing=False, export_labels=False, export_images=True, verbose=False)

        for _, row in quad_log_df[quad_log_df['event'].isin(['low_confidence', 'missing_quad', 'duplicate_quad'])].iterrows():
            warnings.append(f"{row['event']}: {row.get('quad')} (confidence: {row.get('confidence')})")

        result['quadrant_log'] = quad_log_df.to_dict('records')

        # ---- Stage 2: teeth detection per quadrant ----
        teeth_out = os.path.join(session_dir, 'teeth_out')
        teeth_log_df = export_teeth_in_quad_using_enum_model(
            models['enumeration_continued_model'],
            images_root=os.path.join(quad_out, 'images'),
            labels_root=os.path.join(quad_out, 'labels'),
            output_root=teeth_out, conf_threshold=conf_threshold,
            clear_existing=False, export_labels=True, export_images=True, verbose=False)

        for _, row in teeth_log_df[teeth_log_df['event'].isin(
                ['low_confidence', 'missing_teeth', 'needs_manual_review'])].iterrows():
            warnings.append(f"{row['event']}: tooth {row.get('enum_class')} (confidence: {row.get('confidence')})")

        result['teeth_log'] = teeth_log_df.to_dict('records')

        # ---- crop each tooth from the merged labels ----
        all_teeth = []  # flat list across all quadrants, for batched classifier calls
        for fname in os.listdir(os.path.join(teeth_out, 'images')):
            quad_key = os.path.splitext(fname)[0]
            teeth = crop_teeth_from_merged_label(
                os.path.join(teeth_out, 'images', fname),
                os.path.join(teeth_out, 'labels', quad_key + '.txt'),
                class_names['teeth'])
            result['teeth_per_quadrant'][quad_key] = teeth
            for t in teeth:
                t['quad_key'] = quad_key
                all_teeth.append(t)

        if not all_teeth:
            warnings.append("No teeth were detected in any quadrant.")
            return result, warnings

        # ---- Stage 3: healthy vs unhealthy, batched over all teeth at once ----
        dataset = SingleCropDataset([t['image'] for t in all_teeth], CLASSIFIER_TRANSFORM, class_names['healthy_unhealthy'])
        loader = utils.data.DataLoader(dataset, batch_size=16)
        _, preds, probs = predict_classifier(models['teeth_status_model'], loader, device,
                                               class_names['healthy_unhealthy'], return_probs=True)

        diseased_teeth = []
        for t, pred, prob in zip(all_teeth, preds, probs):
            t['status'] = class_names['healthy_unhealthy'][pred]
            t['status_probs'] = dict(zip(class_names['healthy_unhealthy'], prob))
            if t['status'] == 'Disease Found':
                diseased_teeth.append(t)

        if not diseased_teeth:
            result['diseased_teeth'] = []
            return result, warnings

        # ---- Stage 4: disease type, batched over diseased teeth only ----
        dataset = SingleCropDataset([t['image'] for t in diseased_teeth], CLASSIFIER_TRANSFORM, class_names['disease'])
        loader = utils.data.DataLoader(dataset, batch_size=16)
        _, preds, probs = predict_classifier(models['disease_model'], loader, device,
                                               class_names['disease'], return_probs=True)

        caries_teeth = []
        for t, pred, prob in zip(diseased_teeth, preds, probs):
            t['disease'] = class_names['disease'][pred]
            t['disease_probs'] = dict(zip(class_names['disease'], prob))
            if t['disease'] == 'caries':
                caries_teeth.append(t)

        # ---- Stage 5: caries severity, batched over caries teeth only ----
        if caries_teeth:
            dataset = SingleCropDataset([t['image'] for t in caries_teeth], CLASSIFIER_TRANSFORM, class_names['caries_severity'])
            loader = utils.data.DataLoader(dataset, batch_size=16)
            _, preds, probs = predict_classifier(models['caries_status_model'], loader, device,
                                                   class_names['caries_severity'], return_probs=True)
            for t, pred, prob in zip(caries_teeth, preds, probs):
                t['disease'] = class_names['caries_severity'][pred]
                t['caries_probs'] = dict(zip(class_names['caries_severity'], prob))

        result['diseased_teeth'] = diseased_teeth
        return result, warnings

    finally:
        shutil.rmtree(session_dir, ignore_errors=True)