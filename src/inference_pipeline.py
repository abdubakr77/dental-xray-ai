import cv2
import os


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