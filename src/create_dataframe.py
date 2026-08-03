import pandas as pd
from PIL import Image
import os

diagnosis_map = {
    0: "impacted",
    1: "caries",
    2: "periapical",
    3: "deep_caries"
}

quad_map = {
    0: "Upper Right",
    1: "Upper Left",
    2: "Lower Left",
    3: "Lower Right"
}

QUAD_MAP2 = {
    'upperright': 'Upper Right',
    'upperleft': 'Upper Left',
    'lowerright': 'Lower Right',
    'lowerleft': 'Lower Left',
}

def create_diseases_df(disease_json_file,diagnosis_map, quad_map=None, segmentation= False, enumeration=False):

    f_name = []
    bbox = []
    img_h = []
    img_w = []
    disease = []

    if quad_map:
        quad = []
    if segmentation:
        seg = []
    if enumeration:
        tooth_num = []

    for item in disease_json_file['images']:
        for ann in disease_json_file['annotations']:
            if int(ann['image_id']) == int(item['id']):
                f_name.append(item['file_name'])
                bbox.append(ann['bbox'])
                img_h.append(item['height'])
                img_w.append(item['width'])
                disease.append(diagnosis_map[int(ann['category_id_3'])])
                if quad_map:
                    quad.append(quad_map[ann['category_id_1']])
                if segmentation:
                    seg.extend(ann['segmentation'])
                if enumeration:
                    tooth_num.append(ann['category_id_2'])

    data = {
        'File_Name': f_name,
        'Bbox': bbox,
        'Height': img_h,
        'Width': img_w,
        'Disease_Name': disease
    }

    if segmentation:
        data['Seg'] = seg
    if quad_map:
        data['Quad'] = quad
    if enumeration:
        data['Enumeration'] = tooth_num

    return pd.DataFrame(data)


def create_enum_df(enum_json_file, quad_map=None, segmentation= False):

    f_name = []
    bbox = []
    img_h = []
    img_w = []
    tooth_num = []

    if quad_map:
        quad = []
    if segmentation:
        seg = []

    for item in enum_json_file['images']:
        for ann in enum_json_file['annotations']:
            if int(ann['image_id']) == int(item['id']):
                f_name.append(item['file_name'])
                bbox.append(ann['bbox'])
                img_h.append(item['height'])
                img_w.append(item['width'])
                tooth_num.append(ann['category_id_2'])
                if quad_map:
                    quad.append(quad_map[ann['category_id_1']])
                if segmentation:
                    seg.extend(ann['segmentation'])

    data = {
        'File_Name': f_name,
        'Bbox': bbox,
        'Height': img_h,
        'Width': img_w,
        'Enumeration': tooth_num,
    }

    if segmentation:
        data['Seg'] = seg
    if quad_map:
        data['Quad'] = quad

    return pd.DataFrame(data)


def create_quad_df(quad_json_file, quad_map=quad_map, segmentation= False):

    f_name = []
    bbox = []
    img_h = []
    img_w = []
    quad = []

    if segmentation:
        seg = []

    for item in quad_json_file['images']:
        for ann in quad_json_file['annotations']:
            if int(ann['image_id']) == int(item['id']):
                f_name.append(item['file_name'])
                bbox.append(ann['bbox'])
                img_h.append(item['height'])
                img_w.append(item['width'])
                if ann['category_id'] == 0:
                    quad.append(quad_map[1])
                elif ann['category_id'] == 1:
                    quad.append(quad_map[0])
                else:
                    quad.append(quad_map[ann['category_id']])
                if segmentation:
                    seg.extend(ann['segmentation'])

    data = {
        'File_Name': f_name,
        'Bbox': bbox,
        'Height': img_h,
        'Width': img_w,
        'Quad': quad
    }

    if segmentation:
        data['Seg'] = seg

    return pd.DataFrame(data)



def get_quad_from_filename(file_name):
    """
    Pulls the quadrant name out of an image filename, e.g. train_621_LowerLeft.png -> 'Lower Left'.

    Args:
        file_name: the image filename

    Returns:
        str: quadrant name, or None if no quadrant tag is found in the name
    """
    name = os.path.splitext(file_name)[0].lower()
    for key, value in QUAD_MAP2.items():
        if key in name:
            return value
    return None


def build_split_dataframe(images_dir, labels_dir):
    """
    Builds one row per bounding box for every image/label pair in a split.

    Args:
        images_dir: folder with the images for this split
        labels_dir: matching folder with the YOLO .txt label files

    Returns:
        pandas.DataFrame with columns File_Name, Bbox, Height, Width, Enumeration, Quad
    """
    rows = []

    for file_name in os.listdir(images_dir):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + '.txt')
        if not os.path.exists(label_path):
            continue

        with Image.open(os.path.join(images_dir, file_name)) as img:
            width, height = img.size

        quad = get_quad_from_filename(file_name)

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # YOLO format: class_id x_center y_center box_width box_height, all normalized 0-1
                class_id, x_c, y_c, box_w, box_h = line.split()[:5]

                # convert normalized YOLO values to absolute pixel coordinates
                bbox = [
                    float(x_c) * width,
                    float(y_c) * height,
                    float(box_w) * width,
                    float(box_h) * height,
                ]

                rows.append({
                    'File_Name': file_name,
                    'Bbox': bbox,
                    'Height': height,
                    'Width': width,
                    'Enumeration': int(float(class_id)),
                    'Quad': quad,
                })

    return pd.DataFrame(rows, columns=['File_Name', 'Bbox', 'Height', 'Width', 'Enumeration', 'Quad'])


def build_and_export_dataframes(dataset_root, export_dir):
    """
    Builds train/valid/test dataframes and saves each one as a pickle.

    Expects dataset_root to contain train, valid, and test folders,
    each with an images/ and a labels/ subfolder.

    Args:
        dataset_root: path containing the train, valid, and test folders
        export_dir: folder to save the resulting pickle files into

    Returns:
        dict with keys 'train', 'valid', 'test' mapping to their dataframes
    """
    os.makedirs(export_dir, exist_ok=True)
    dataframes = {}

    for split_name in ['train', 'valid', 'test']:
        images_dir = os.path.join(dataset_root, split_name, 'images')
        labels_dir = os.path.join(dataset_root, split_name, 'labels')

        df = build_split_dataframe(images_dir, labels_dir)
        dataframes[split_name] = df

        export_path = os.path.join(export_dir, f'{split_name}_df.pkl')
        df.to_pickle(export_path)
        print(f'{split_name}: {len(df)} rows -> {export_path}')

    return dataframes