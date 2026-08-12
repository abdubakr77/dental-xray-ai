import cv2
import os
from torch import utils

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