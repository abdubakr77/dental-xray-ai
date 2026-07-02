import pandas as pd

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
                quad.append(quad_map[ann['category_id_1']])
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