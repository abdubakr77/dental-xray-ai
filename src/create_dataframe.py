import pandas as pd

def create_dentex_df(coco_json_file,diseases_dict, quadrant_dict=None, segmentation= False, enumeration=False):

    f_name = []
    bbox = []
    img_h = []
    img_w = []
    disease = []

    if quadrant_dict:
        quad = []
    if segmentation:
        seg = []
    if enumeration:
        tooth_num = []

    for item in coco_json_file['images']:
        for ann in coco_json_file['annotations']:
            if int(ann['image_id']) == int(item['id']):
                f_name.append(item['file_name'])
                bbox.append(ann['bbox'])
                img_h.append(item['height'])
                img_w.append(item['width'])
                disease.append(diseases_dict[int(ann['category_id_3'])])
                if quadrant_dict:
                    quad.append(quadrant_dict[ann['category_id_1']])
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

    if quadrant_dict:
        data['Seg'] = seg
    if segmentation:
        data['Quad'] = quad
    if enumeration:
        data['Tooth_Num'] = tooth_num

    return pd.DataFrame(data)