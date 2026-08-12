import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from src.utils import _iou_xyxy,_xywh_norm_to_xyxy_px
from src.vis import visualize_augmentation
from tqdm import tqdm
import pandas as pd
from ultralytics import YOLO
from itertools import combinations
from pathlib import Path
import shutil
import random
import torch.nn as nn
from torch import exp

def draw_corner_box(img, x1, y1, x2, y2, label_name, confidence, color, length, thickness):
    """
    Draw a corner-style bounding box with a label badge on an image in-place.

    Args:
        img        : RGB numpy array to draw on (modified in place)
        x1, y1    : top-left corner in pixels
        x2, y2    : bottom-right corner in pixels
        label_name : class label string
        confidence : confidence score in [0, 1]
        color      : BGR colour tuple for the box and badge
        length     : length of each corner tick in pixels
        thickness  : line thickness in pixels
    """
    # draw L-shaped ticks at each corner
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

# draw filled badge above the box, then write text on top
    if label_name and confidence:
        text = f"{label_name} {confidence*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
        cv2.putText(img, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)


def dedupe_gt_label_files(labels_root, iou_threshold=0.85, dry_run=True, verbose=False):
    """
    Remove exact and near-duplicate lines within each YOLO label file in
    labels_root. This is a data-cleaning pass on the disease labels
    themselves -- it doesn't touch images, doesn't run any model, and has
    nothing to do with export_teeth_in_quad_using_enum_model beyond being a
    sensible thing to run before it.

    A tooth can genuinely carry more than one disease box -- different
    conditions, visibly different box sizes or positions -- and that stays
    untouched. What this removes is the other case: two lines with the same
    class whose boxes are nearly identical (IoU >= iou_threshold), which
    looks like the same annotation exported twice rather than two separate
    findings. In the file that flagged this (train_481_LowerRight.txt), one
    pair was byte-for-byte identical (IoU 1.0) and one pair was IoU ~0.91 --
    same box, tiny floating-point drift between two export runs. The default
    threshold sits below that to catch both while staying well clear of
    genuinely different disease boxes on the same tooth, which look nothing
    alike in size.

    Args:
        labels_root: folder of YOLO label .txt files to clean
        iou_threshold: IoU above which two same-class lines in the same file
            count as duplicates of each other
        dry_run: if True (default), only reports what WOULD be removed --
            no file is touched. Run this first, check the numbers, then
            call again with dry_run=False to actually clean the files.
        verbose: print each duplicate pair as it's found

    Returns:
        a DataFrame with one row per file that had duplicates:
        File_Name, n_duplicates_removed
    """
    report_records = []
    label_files = [f for f in os.listdir(labels_root) if f.lower().endswith('.txt')]

    for fname in tqdm(label_files):
        path = os.path.join(labels_root, fname)
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        parsed = []
        for line in lines:
            parts = line.split()
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            # scale doesn't matter for IoU as long as it's consistent, so this
            # skips opening the image entirely -- just work in normalized space
            x1, y1, x2, y2 = _xywh_norm_to_xyxy_px(cx, cy, w, h, 1.0, 1.0)
            parsed.append({'line': line, 'cls_id': cls_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        keep = [True] * len(parsed)
        n_removed = 0

        for i in range(len(parsed)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(parsed)):
                if not keep[j] or parsed[j]['cls_id'] != parsed[i]['cls_id']:
                    continue
                iou = _iou_xyxy((parsed[i]['x1'], parsed[i]['y1'], parsed[i]['x2'], parsed[i]['y2']),
                                 (parsed[j]['x1'], parsed[j]['y1'], parsed[j]['x2'], parsed[j]['y2']))
                if iou >= iou_threshold:
                    keep[j] = False
                    n_removed += 1
                    if verbose:
                        print(f"{fname}: duplicate tooth {parsed[i]['cls_id']} (IoU={iou:.3f}), removing one copy")

        if n_removed > 0:
            report_records.append({'File_Name': fname, 'n_duplicates_removed': n_removed})
            if not dry_run:
                with open(path, 'w') as f:
                    for p, k in zip(parsed, keep):
                        if k:
                            f.write(p['line'] + "\n")

    report_df = pd.DataFrame(report_records)
    print(f"Files with duplicates: {len(report_df)} out of {len(label_files)}")
    if len(report_df) > 0:
        print(f"Total duplicate lines removed: {report_df['n_duplicates_removed'].sum()}")
    if dry_run:
        print("dry_run=True -- no files were changed. Re-run with dry_run=False to apply.")

    return report_df


def smart_predict(yolo_model, images_path, specific_image_name=None, conf_threshold=0.3, 
                  show_true_boxes = False, save_crop_output_image:str=False, 
                  save_output:bool=False, save_dir:str=None, 
                  apply_custom_draw_box=False,color=(255, 0, 0), length=150, thickness=5):

    if not os.path.exists(images_path):
        raise FileNotFoundError('Image Path not existed! Please Check the path is correct')
    
    rand_image_name = np.random.choice(os.listdir(images_path)).split('.')[0]
    image_path = os.path.join(images_path,rand_image_name+'.png')

    if specific_image_name:
        rand_image_name = specific_image_name.split('.')[0]
        image_path = os.path.join(images_path,rand_image_name+'.png')


    outputs = yolo_model.predict(image_path,conf=conf_threshold)

    output = outputs[0]
    boxes = output.boxes
    names = output.names

    original_image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)

    image_custom_draw = original_image.copy()

    _,ax = plt.subplots(1,2,figsize=(18,12))
    for i in range(len(boxes)):
        confidence = np.round(boxes.conf[i].item(),2)
        coordinates = boxes.xyxy[i].tolist()
        cls_name = names[boxes.cls[i].item()]

        x1, y1, x2, y2 = map(int, coordinates)
        draw_corner_box(image_custom_draw, x1, y1, x2, y2, cls_name, confidence,color,length,thickness)

        if save_crop_output_image:
            cropped_image = original_image[y1:y2, x1:x2]
            if os.path.exists(save_dir):
                cv2.imwrite(os.path.join(save_dir,f'{rand_image_name}_{i}_{cls_name}.png'),cropped_image)
            else:
                raise Exception('Save Dir Path is not existed! Please check the path is correct and exists')

        print(f'Class Name: {cls_name}')
        print(f'Coordinates: {coordinates}')
        print(f'Confidence: %{(confidence * 100):.4}')

    ax[0].set_title('Original Image')
    ax[0].imshow(original_image)
    ax[0].axis('off')

    ax[1].set_title('Object Detected')

    if apply_custom_draw_box:
        output_image = image_custom_draw
    else:
        output_image = output.plot()[:,:,::-1]

    if show_true_boxes:
        image_h , image_w = original_image.shape[:2]
        labels_path = images_path.replace('images','labels')
        label_name = rand_image_name
        label_file_path = os.path.join(labels_path,label_name+'.txt')
        with open(label_file_path,'r') as f:
            for line in f.readlines():

                cx,cy,w,h = map(float,line.split()[1:])

                x1 = int((cx - w/2) * image_w)
                y1 = int((cy - h/2) * image_h)
                x2 = int((cx + w/2) * image_w)
                y2 = int((cy + h/2) * image_h)
                
                draw_corner_box(output_image if apply_custom_draw_box else output_image[:,:,::-1], x1, y1, x2, y2, None, None ,color=(0, 255, 0),length=50,thickness=2)

    ax[1].imshow(output_image)
    ax[1].axis('off')
    
    if save_output:
        if os.path.exists(save_dir):
            cv2.imwrite(os.path.join(save_dir,f'{rand_image_name}_full_output.png'),output_image)
        else:
            raise Exception('Save Dir Path is not existed! Please check the path is correct and exists')
        

    plt.tight_layout()
    plt.show()


def export_quadrants_using_quad_model(quadrant_model, original_images_path, annotations_df=None,
                                    output_root=os.getcwd(),
                                    conf_threshold=0.3,
                                    debugging=False, debug_limit=5,
                                    clear_existing=True,
                                    verbose=False,
                                    export_labels=True,
                                    export_images=True,
                                    specific_image_name=None):


    os.makedirs(output_root, exist_ok=True)

    if export_images:
        os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)

    if export_labels:
        os.makedirs(os.path.join(output_root, 'labels'), exist_ok=True)

    # ---- check + clear existing images/labels ----
    if clear_existing and not debugging:
        imgs_path = os.path.join(output_root, 'images')
        labels_path = os.path.join(output_root, 'labels')

        any_existing = (export_images and os.path.exists(imgs_path) and len(os.listdir(imgs_path)) > 0) or \
                        (export_labels and os.path.exists(labels_path) and len(os.listdir(labels_path)) > 0)

        if any_existing:
            print("Warning: Found existing images/labels in the output directory.")
            confirm = input("Do you want to delete them all before re-exporting? - (y or n): ").lower().strip()

            if confirm == 'y':
                deleted_count = 0
                failed_count = 0
                paths_to_clear = []
                if export_images:
                    paths_to_clear.append(imgs_path)
                if export_labels:
                    paths_to_clear.append(labels_path)
                for sub_path in paths_to_clear:
                    if not os.path.exists(sub_path):
                        continue
                    for f in os.listdir(sub_path):
                        try:
                            os.remove(os.path.join(sub_path, f))
                            deleted_count += 1
                        except Exception as e:
                            print(f"Failed to remove {f}: {e}")
                            failed_count += 1

                print(f"Deleted {deleted_count} files. Failed: {failed_count}.")
            else:
                print("Skipped clearing. New files will be mixed with existing ones.")

    debug_count = 0
    all_images, all_labels, all_bboxes, all_filenames = [], [], [], []
    log_records = []

    file_list = annotations_df['File_Name'].unique()

    if debugging:
        
        if len(specific_image_name) >= 1 and type(specific_image_name) == list:
            file_list = specific_image_name
        else:
            file_list = [np.random.choice(file_list)]

    for fname in tqdm(file_list):

        img_path = os.path.join(original_images_path, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = quadrant_model.predict(img_path, conf=conf_threshold, verbose=False)[0]
        n_predicted_boxes = len(results.boxes)
        expected_quads = ["Upper Right", "Upper Left", "Lower Left", "Lower Right"]
        processed_quads_this_image = set()

        for box in results.boxes:

            quad_class_id = int(box.cls[0])
            quad_name = quadrant_model.names[quad_class_id]
            confidence = np.round(float(box.conf), 2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            crop_h, crop_w = cropped_img.shape[:2]
            if crop_h == 0 or crop_w == 0:
                continue

            new_labels = []
            if export_labels:
                teeth_rows = annotations_df[(annotations_df['File_Name'] == fname) & (annotations_df['Quad'] == quad_name)]

                for _, row in teeth_rows.iterrows():
                    tx, ty, tw, th = list(row['Bbox'])
                    new_x = tx - x1
                    new_y = ty - y1

                    clipped_x1 = max(new_x, 0)
                    clipped_y1 = max(new_y, 0)
                    clipped_x2 = min(new_x + tw, crop_w)
                    clipped_y2 = min(new_y + th, crop_h)

                    clipped_w = clipped_x2 - clipped_x1
                    clipped_h = clipped_y2 - clipped_y1

                    if clipped_w <= 0 or clipped_h <= 0:
                        continue

                    cx = (clipped_x1 + clipped_w / 2) / crop_w
                    cy = (clipped_y1 + clipped_h / 2) / crop_h
                    nw = clipped_w / crop_w
                    nh = clipped_h / crop_h

                    new_labels.append((row['Enumeration'], cx, cy, nw, nh))

                if not new_labels:
                    continue

            base_name = f"{fname.split('.')[0]}_{quad_name.replace(' ', '')}"

            if debugging:
                all_images.append(cropped_img)
                all_bboxes.append([(cx, cy, nw, nh) for _, cx, cy, nw, nh in new_labels])
                all_labels.append([cls_id for cls_id, _, _, _, _ in new_labels])
                all_filenames.append(base_name)

                debug_count += 1
                if debug_count >= debug_limit:
                    break
                continue

            else:
                img_out_path = os.path.join(output_root, 'images', f"{base_name}.png")
                label_out_path = os.path.join(output_root, 'labels', f"{base_name}.txt")

                is_duplicate = quad_name not in expected_quads

                if is_duplicate:
                    log_records.append({
                        'File_Name': fname, 'event': 'duplicate_quad',
                        'quad': quad_name, 'confidence': confidence,
                        'n_boxes': n_predicted_boxes,
                        'crop_area': crop_h * crop_w
                    })
                    if verbose:
                        print(f'Warning: Model Detected {n_predicted_boxes} Boxes And Predicted {quad_name} Again With {confidence} Confidence In This Image Name: {fname.split(".")[0]}')
                    
                    if quad_name in processed_quads_this_image:
                        if verbose:
                            print("Skipped Successfuly!")
                        continue

                        

                expected_quads.remove(quad_name) if quad_name in expected_quads else None
                processed_quads_this_image.add(quad_name)

                if confidence < 0.6:
                    log_records.append({
                        'File_Name': fname, 'event': 'low_confidence',
                        'quad': quad_name, 'confidence': confidence,
                        'n_boxes': n_predicted_boxes,
                        'crop_area': crop_h * crop_w
                    })
                    if verbose:
                        print(f"Warning: Low Confidence Alert! Got {confidence} At Quadrant {quad_name} In Image Name: {fname.split('.')[0]}")

                log_records.append({
                    'File_Name': fname, 'event': 'successful_detection',
                    'quad': quad_name, 'confidence': confidence,
                    'n_boxes': n_predicted_boxes,
                    'crop_area': crop_h * crop_w
                })

                if export_images:
                    cv2.imwrite(img_out_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

                if export_labels:
                    with open(label_out_path, 'w') as f:
                        for cls_id, cx, cy, nw, nh in new_labels:
                            f.write(f"{cls_id} {cx} {cy} {nw} {nh}\n")

        log_records.append({
            'File_Name': fname, 'event': 'successful_detection',
            'quad': quad_name, 'confidence': confidence,
            'n_boxes': n_predicted_boxes,
            'crop_area': crop_h * crop_w,
            'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
        })

        if expected_quads:
            log_records.append({
                'File_Name': fname, 'event': 'missing_quad',
                'quad': ' & '.join(expected_quads), 'confidence': None,
                'n_boxes': n_predicted_boxes,
                'crop_area': None
            })
            if verbose:
                missing_str = ' & '.join(expected_quads) if len(expected_quads) >= 2 else expected_quads[0]
                print(f"Warning: Model Detected {n_predicted_boxes} Boxes And Can't Predict {missing_str} In This Image Name: {fname.split('.')[0]}")

        if debugging and debug_count >= debug_limit:
            visualize_augmentation(all_images, all_bboxes, all_labels, titles=all_filenames)
            break
    if not debugging:
        log_df = pd.DataFrame(log_records)
        return log_df


def export_teeth_in_quad_using_enum_model(enum_model, images_root, labels_root,
                                            output_root=os.getcwd(),
                                            conf_threshold=0.3,
                                            low_conf_threshold=0.6,
                                            duplicate_iou_threshold=0.5,
                                            edge_margin_ratio=0.06,
                                            leak_area_ratio=0.5,
                                            background_area_ratio_low=0.15,
                                            background_area_ratio_high=4.0,
                                            merged_box_width_ratio=1.6,
                                            n_enum_classes=8,
                                            debugging=False, debug_limit=5,
                                            clear_existing=True,
                                            verbose=False,
                                            export_labels=True,
                                            export_images=True,
                                            specific_image_name:list=None,
                                            skip_files: list=None):
    """
    Stage 2 Continued step: run the enumeration model on the quadrant crops that
    currently only have disease-tooth labels, and fill in the rest of the teeth.
 
    Unlike export_quadrants_using_quad_model, this does not crop anything itself.
    images_root/labels_root already hold the quadrant crops and their YOLO labels
    (exported earlier via export_quadrants_using_quad_model on the disease data), and
    those labels only cover the diseased teeth. For each crop we run the enum
    model, drop any prediction that lands on a tooth number already present in the
    existing label (that tooth is already correct ground truth, so we never let a
    prediction override it), clean up duplicate detections, check the surviving
    boxes against each other geometrically, and write out a merged label file
    with the original disease teeth plus the newly detected healthy ones.
 
    Two kinds of duplicates get resolved before anything else: the same tooth
    number predicted twice (same-class, handled as before -- same location vs
    different location), and two DIFFERENT tooth numbers predicted on the same
    physical spot (cross-class overlap, new). GT always wins a cross-class
    overlap; among predictions the highest confidence wins.
 
    After that, the surviving boxes (GT + predictions) get sorted geometrically
    by their x position -- tooth 0 sits at the midline, so for a "Right"
    quadrant crop that's the right edge (class increases as x decreases), and
    for a "Left" quadrant crop that's the left edge (class increases as x
    increases). The GT boxes are 100% correct, so they act as anchors: any
    predicted box whose class doesn't fall between its nearest GT neighbors in
    that order gets flagged. A box noticeably wider than the others in the same
    crop (merged_box_width_ratio) gets flagged too, since that usually means two
    teeth got boxed as one and splitting it isn't something this function can
    do on its own.
 
    None of this tries to fully reconstruct a correct labeling by itself --
    when teeth are missing (extracted, unerupted) at unknown positions, the
    true count in a crop is genuinely ambiguous from geometry alone. What it
    does reliably is separate internally-consistent crops from ones that
    aren't, via the 'needs_manual_review' event in log_df, so you can look at
    exactly those before deciding what to keep or delete.
 
    The leak/background checks below are still heuristics, same caveat as
    before -- tune edge_margin_ratio, leak_area_ratio, background_area_ratio_low/high
    against your own data before trusting them blindly.
 
    Args:
        enum_model: loaded YOLO enumeration model
        images_root: folder of quadrant crop images (e.g. Stage 3 crops)
        labels_root: folder of existing YOLO label .txt files for those crops,
            currently containing only the diseased-tooth boxes
        output_root: where the merged images/labels get written
        conf_threshold: passed straight to enum_model.predict
        low_conf_threshold: predictions below this get flagged as low_confidence
            but are still kept
        duplicate_iou_threshold: IoU above which two boxes count as the same
            physical tooth -- used both for same-class duplicates and for the
            cross-class overlap check
        edge_margin_ratio: how close to the crop edge (as a fraction of width or
            height) a box has to sit before it's considered for the cross-quadrant
            leak check
        leak_area_ratio: a box touching the edge is only flagged as a possible
            leak if its area is below this fraction of the median accepted box
            area for the image
        background_area_ratio_low / _high: a box is flagged as a possible
            background prediction if its area falls outside
            [median * low, median * high] for that image
        merged_box_width_ratio: a box this many times wider than the median
            box in its crop gets flagged as a possible two-teeth-in-one detection
        n_enum_classes: number of enumeration classes (0-7 by default, matching
            the project's data.yaml)
        debugging: if True, sample debug_limit random crops and visualize instead
            of writing anything out
        clear_existing: if True (and not debugging), ask before wiping
            output_root/images and output_root/labels
        verbose: print a line for every flagged event as it happens
        export_labels / export_images: toggle writing each output type
 
    Returns:
        log_df: a DataFrame of one row per event. None when debugging=True.
    """


    os.makedirs(output_root, exist_ok=True)

    if export_images:
        os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)

    if export_labels:
        os.makedirs(os.path.join(output_root, 'labels'), exist_ok=True)

 
    # ---- check + clear existing images/labels ----
    if clear_existing and not debugging:
        imgs_path = os.path.join(output_root, 'images')
        labels_path = os.path.join(output_root, 'labels')
 
        any_existing = (export_images and os.path.exists(imgs_path) and len(os.listdir(imgs_path)) > 0) or \
                        (export_labels and os.path.exists(labels_path) and len(os.listdir(labels_path)) > 0)
 
        if any_existing:
            print("Warning: Found existing images/labels in the output directory.")
            confirm = input("Do you want to delete them all before re-exporting? - (y or n): ").lower().strip()
 
            if confirm == 'y':
                deleted_count = 0
                failed_count = 0
                paths_to_clear = []
                if export_images:
                    paths_to_clear.append(imgs_path)
                if export_labels:
                    paths_to_clear.append(labels_path)
                for sub_path in paths_to_clear:
                    if not os.path.exists(sub_path):
                        continue
                    for f in os.listdir(sub_path):
                        try:
                            os.remove(os.path.join(sub_path, f))
                            deleted_count += 1
                        except Exception as e:
                            print(f"Failed to remove {f}: {e}")
                            failed_count += 1
 
                print(f"Deleted {deleted_count} files. Failed: {failed_count}.")
            else:
                print("Skipped clearing. New files will be mixed with existing ones.")
 
    if export_images:
        os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
    if export_labels:
        os.makedirs(os.path.join(output_root, 'labels'), exist_ok=True)
 
    file_list = [f for f in os.listdir(images_root) if f.lower().endswith('.png')]
 
    if debugging:
        
        if len(specific_image_name) >= 1 and type(specific_image_name) == list:
            file_list = specific_image_name
        else:
            file_list = [np.random.choice(file_list)]
    
    debug_count = 0
    all_images, all_labels, all_bboxes, all_filenames = [], [], [], []
    log_records = []
 
    # The four crop filename suffixes produced by export_quadrants_using_quad_model,
    # e.g. "1234_UpperRight.png". Used here to recover which quadrant a crop belongs
    # to, since the enum model itself has no notion of quadrant.
    QUAD_TAGS = ["UpperRight", "UpperLeft", "LowerLeft", "LowerRight"]
 
    for fname in tqdm(file_list):
 
        name_no_ext = os.path.splitext(fname)[0]
 
        if skip_files is not None and fname in skip_files:
            print(f'Bad Image {fname} Skipped Successfuly!')
            continue

        # recover which quadrant this crop came from, from the filename suffix
        # export_quadrants_using_quad_model gave it (e.g. "..._UpperRight")
        quad_name = None
        for tag in QUAD_TAGS:
            if name_no_ext.endswith('_' + tag):
                quad_name = tag
                break
 
        img_path = os.path.join(images_root, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_h, crop_w = img.shape[:2]
 
        # ---- load the existing (disease-only) labels for this crop, if any ----
        # now keeping pixel coordinates too, needed for the overlap and
        # ordering checks below (used to just track class ids)
        existing_lines = []
        existing_classes = set()
        existing_boxes = []
        label_path = os.path.join(labels_root, f"{name_no_ext}.txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    ex1, ey1, ex2, ey2 = _xywh_norm_to_xyxy_px(cx, cy, w, h, crop_w, crop_h)
                    existing_lines.append(line.strip())
                    existing_classes.add(cls_id)
                    existing_boxes.append({'cls_id': cls_id, 'x1': ex1, 'y1': ey1, 'x2': ex2, 'y2': ey2,
                                            'cx': cx, 'cy': cy, 'nw': w, 'nh': h})
 
        # ---- run the enum model on the crop ----
        results = enum_model.predict(img_path, conf=conf_threshold, verbose=False)[0]
        n_predicted_boxes = len(results.boxes)
 
        # group raw predictions by class so we can resolve same-class duplicates first
        by_class = {}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = np.round(float(box.conf), 2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
 
            # a tooth number already present in the existing (disease) labels is
            # already correct ground truth, so we never let a prediction touch it.
            # This isn't a duplicate in the usual sense, so it gets its own event.
            if cls_id in existing_classes:
                log_records.append({
                    'File_Name': fname, 'event': 'skipped_already_labeled',
                    'enum_class': cls_id, 'confidence': confidence,
                    'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
                })
                if verbose:
                    print(f"Skipped: tooth {cls_id} already has a disease label in {fname}")
                continue
 
            by_class.setdefault(cls_id, []).append((confidence, (x1, y1, x2, y2)))
 
        accepted_boxes = []  # list of (cls_id, confidence, x1, y1, x2, y2)
 
        for cls_id, candidates in by_class.items():
            if len(candidates) == 1:
                confidence, box = candidates[0]
                accepted_boxes.append((cls_id, confidence, *box))
                continue

            # more than one box predicted the same tooth number: figure out if
            # they're really the same detection duplicated (high IoU) or two
            # genuinely different locations (model confused about where this
            # tooth number is).
            max_iou = max(_iou_xyxy(a[1], b[1]) for a, b in combinations(candidates, 2))
            dup_event = 'duplicate_same_location' if max_iou >= duplicate_iou_threshold else 'duplicate_diff_location'

            # rank all candidates by confidence first, as before
            ranked_indices = sorted(range(len(candidates)), key=lambda i: candidates[i][0], reverse=True)

            # if the candidates sit at different physical spots, one of them may be a
            # leak from the neighboring row (e.g. an upper tooth showing up in a
            # Lower crop). Prefer whichever candidate's vertical position matches this
            # crop's own quadrant side before falling back to confidence alone.
            if dup_event == 'duplicate_diff_location' and quad_name is not None:
                def matches_quadrant_side(box):
                    y1, y2 = box[1], box[3]
                    cy = (y1 + y2) / 2
                    if 'Upper' in quad_name:
                        return cy <= crop_h / 2
                    if 'Lower' in quad_name:
                        return cy >= crop_h / 2
                    return True

                matching_indices = [i for i in ranked_indices if matches_quadrant_side(candidates[i][1])]
                if matching_indices:
                    ranked_indices = matching_indices + [i for i in ranked_indices if i not in matching_indices]

            best_idx = ranked_indices[0]
            best_conf, best_box = candidates[best_idx]
            accepted_boxes.append((cls_id, best_conf, *best_box))

            for i, (confidence, box) in enumerate(candidates):
                if i == best_idx:
                    continue
                log_records.append({
                    'File_Name': fname, 'event': dup_event,
                    'enum_class': cls_id, 'confidence': confidence,
                    'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
                })
                if verbose:
                    print(f"Warning: dropped a {dup_event} box for tooth {cls_id} in {fname}")
 
        # ---- cross-class overlap resolution ----
        kept_candidates = [{'cls_id': b['cls_id'], 'x1': b['x1'], 'y1': b['y1'], 'x2': b['x2'], 'y2': b['y2'],
                             'confidence': None, 'is_gt': True} for b in existing_boxes]

        pred_candidates = sorted(
            [{'cls_id': cls_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'confidence': confidence, 'is_gt': False}
             for cls_id, confidence, x1, y1, x2, y2 in accepted_boxes],
            key=lambda b: -b['confidence']
        )

        for cand in pred_candidates:
            cand_xyxy = (cand['x1'], cand['y1'], cand['x2'], cand['y2'])
            overlaps_kept = any(_iou_xyxy(cand_xyxy, (k['x1'], k['y1'], k['x2'], k['y2'])) >= duplicate_iou_threshold
                                 for k in kept_candidates)
            if overlaps_kept:
                log_records.append({
                    'File_Name': fname, 'event': 'cross_class_overlap',
                    'enum_class': cand['cls_id'], 'confidence': cand['confidence'],
                    'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
                })
                if verbose:
                    print(f"Warning: dropped overlapping box (class {cand['cls_id']}, cross_class_overlap) in {fname}")
                continue
            kept_candidates.append(cand)
 
        # only the surviving predictions go back into accepted_boxes; existing_boxes
        # (GT) is untouched by design
        accepted_boxes = [(c['cls_id'], c['confidence'], c['x1'], c['y1'], c['x2'], c['y2'])
                           for c in kept_candidates if not c['is_gt']]
 
        # median accepted box area for this image, used by the leak/background checks below
        areas = [max(0, x2 - x1) * max(0, y2 - y1) for _, _, x1, y1, x2, y2 in accepted_boxes]
        median_area = float(np.median(areas)) if areas else 0.0
 
        final_boxes = []     # (cls_id, cx, cy, nw, nh) normalized, what gets written out
        final_boxes_px = []  # same boxes in pixels, needed for the ordering/merged checks below
 
        for cls_id, confidence, x1, y1, x2, y2 in accepted_boxes:
            box_w, box_h = x2 - x1, y2 - y1
            box_area = box_w * box_h
 
            # --- possible leak from a neighboring quadrant ---
            # heuristic: the box sits right at the crop edge away from the tooth
            # row AND is much smaller than the other accepted teeth in this crop,
            # which is what a partially cropped, leaked-in tooth tends to look like
            is_edge_touching = False
            if quad_name is not None:
                edge_x = edge_margin_ratio * crop_w
                edge_y = edge_margin_ratio * crop_h
                if 'Upper' in quad_name and y2 >= crop_h - edge_y:
                    is_edge_touching = True
                if 'Lower' in quad_name and y1 <= edge_y:
                    is_edge_touching = True
                if 'Right' in quad_name and x2 >= crop_w - edge_x:
                    is_edge_touching = True
                if 'Left' in quad_name and x1 <= edge_x:
                    is_edge_touching = True
 
            if is_edge_touching and median_area > 0 and box_area <= leak_area_ratio * median_area:
                log_records.append({
                    'File_Name': fname, 'event': 'possible_cross_quadrant_leak',
                    'enum_class': cls_id, 'confidence': confidence,
                    'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
                })
                if verbose:
                    print(f"Warning: possible cross-quadrant leak for tooth {cls_id} in {fname}")
                continue
 
            # --- possible background prediction ---
            # heuristic: a box wildly smaller or larger than the other teeth in
            # this same crop is more likely to be background than a real tooth
            if median_area > 0 and not (background_area_ratio_low * median_area <= box_area <= background_area_ratio_high * median_area):
                log_records.append({
                    'File_Name': fname, 'event': 'possible_background_prediction',
                    'enum_class': cls_id, 'confidence': confidence,
                    'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
                })
                if verbose:
                    print(f"Warning: possible background prediction for tooth {cls_id} in {fname}")
                continue
 
            if confidence < low_conf_threshold:
                log_records.append({
                    'File_Name': fname, 'event': 'low_confidence',
                    'enum_class': cls_id, 'confidence': confidence,
                    'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
                })
                if verbose:
                    print(f"Warning: low confidence ({confidence}) for tooth {cls_id} in {fname}")
 
            log_records.append({
                'File_Name': fname, 'event': 'successful_detection',
                'enum_class': cls_id, 'confidence': confidence,
                'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
            })
 
            cx = ((x1 + x2) / 2) / crop_w
            cy = ((y1 + y2) / 2) / crop_h
            nw = box_w / crop_w
            nh = box_h / crop_h
            final_boxes.append((cls_id, cx, cy, nw, nh))
            final_boxes_px.append((cls_id, confidence, x1, y1, x2, y2))
 
        # ---- geometric ordering against GT anchors (new) ----
        # tooth 0 sits at the midline: Right quadrant -> crop's right edge,
        # class increases as x decreases. Left quadrant -> crop's left edge,
        # class increases as x increases. GT boxes are trusted anchors; a
        # predicted box whose class falls outside its neighboring anchors'
        # range gets flagged rather than silently kept.
        needs_review = False
        review_reasons = []
 
        combined_px = [{'cls_id': b['cls_id'], 'x1': b['x1'], 'x2': b['x2'], 'is_gt': True}
                       for b in existing_boxes]
        combined_px += [{'cls_id': cls_id, 'x1': x1, 'x2': x2, 'is_gt': False}
                        for cls_id, confidence, x1, y1, x2, y2 in final_boxes_px]
 
        if quad_name is None:
            needs_review = True
            review_reasons.append("could not determine quadrant side from filename, ordering not checked")
        else:
            reverse = 'Right' in quad_name
            combined_sorted = sorted(combined_px, key=lambda b: (b['x1'] + b['x2']) / 2, reverse=reverse)
            anchor_positions = [(i, b['cls_id']) for i, b in enumerate(combined_sorted) if b['is_gt']]
 
            for i, b in enumerate(combined_sorted):
                if b['is_gt']:
                    continue
                left_anchor = max([c for pos, c in anchor_positions if pos < i], default=None)
                right_anchor = min([c for pos, c in anchor_positions if pos > i], default=None)
                lo = left_anchor if left_anchor is not None else -1
                hi = right_anchor if right_anchor is not None else n_enum_classes
                if not (lo <= b['cls_id'] <= hi):
                    needs_review = True
                    review_reasons.append(f"tooth {b['cls_id']} conflicts with anchors (expected class between {lo} and {hi})")
                    log_records.append({
                        'File_Name': fname, 'event': 'ordering_inconsistent',
                        'enum_class': b['cls_id'], 'confidence': None,
                        'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
                    })
 
        # ---- possible merged box: one box spanning two teeth (new) ----
        widths = [b['x2'] - b['x1'] for b in combined_px]
        median_width = float(np.median(widths)) if widths else 0.0
        for cls_id, confidence, x1, y1, x2, y2 in final_boxes_px:
            box_w = x2 - x1
            if median_width > 0 and box_w >= merged_box_width_ratio * median_width:
                needs_review = True
                review_reasons.append(f"tooth {cls_id} box is {box_w / median_width:.1f}x the median width, may span two teeth")
                log_records.append({
                    'File_Name': fname, 'event': 'possible_merged_box',
                    'enum_class': cls_id, 'confidence': confidence,
                    'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
                })
 
        # not every quadrant truly has all n_enum_classes teeth (missing/extracted
        # teeth are normal), so this is a flag to review, not a guaranteed error
        covered_classes = existing_classes | {cls_id for cls_id, *_ in final_boxes}
        missing_classes = set(range(n_enum_classes)) - covered_classes
        if missing_classes:
            log_records.append({
                'File_Name': fname, 'event': 'missing_teeth',
                'enum_class': ' & '.join(str(c) for c in sorted(missing_classes)),
                'confidence': None, 'n_boxes': n_predicted_boxes, 'crop_area': None
            })
            if verbose:
                print(f"Warning: no detection for teeth {sorted(missing_classes)} in {fname}")
 
        log_records.append({
            'File_Name': fname, 'event': 'boxes_detected',
            'enum_class': None, 'confidence': None,
            'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
        })
 
        if needs_review:
            log_records.append({
                'File_Name': fname, 'event': 'needs_manual_review',
                'enum_class': ' | '.join(review_reasons), 'confidence': None,
                'n_boxes': n_predicted_boxes, 'crop_area': crop_h * crop_w
            })
            if verbose:
                print(f"Flagged for review: {fname} -> {'; '.join(review_reasons)}")
 
        if debugging:
            # pull labels and boxes from the SAME ordered list (existing_boxes) for the
            # GT portion, instead of a list zipped against a set
            debug_labels = [cls_id for cls_id, *_ in final_boxes] + [b['cls_id'] for b in existing_boxes]
            debug_bboxes = [(cx, cy, nw, nh) for _, cx, cy, nw, nh in final_boxes] + \
                            [(b['cx'], b['cy'], b['nw'], b['nh']) for b in existing_boxes]
 
            all_images.append(img)
            all_bboxes.append(debug_bboxes)
            all_labels.append(debug_labels)
            all_filenames.append(name_no_ext + (' [REVIEW]' if needs_review else ''))
 
            debug_count += 1
            if debug_count >= debug_limit:
                visualize_augmentation(all_images, all_bboxes, all_labels, titles=all_filenames)
                break
            continue
 
        # ---- write merged output: existing disease labels + newly accepted teeth ----
        if export_images:
            cv2.imwrite(os.path.join(output_root, 'images', fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
 
        if export_labels:
            with open(os.path.join(output_root, 'labels', f"{name_no_ext}.txt"), 'w') as f:
                for line in existing_lines:
                    f.write(line + "\n")
                for cls_id, cx, cy, nw, nh in final_boxes:
                    f.write(f"{cls_id} {cx} {cy} {nw} {nh}\n")
 
    if not debugging:
        log_df = pd.DataFrame(log_records)
        return log_df
 
 
def get_review_candidates(log_df,return_df=False):
    """
    Pull the images flagged during export_teeth_in_quad_using_enum_model for
    manual review, with the reasons attached. Meant to be looked at and
    decided on by hand, not fed into an automatic delete step.
 
    Args:
        log_df: the DataFrame returned by export_teeth_in_quad_using_enum_model
 
    Returns:
        a DataFrame with one row per flagged image: File_Name and reasons
    """
    review_df = log_df[log_df['event'] == 'needs_manual_review'][['File_Name', 'enum_class']].copy()
    review_df = review_df.rename(columns={'enum_class': 'reasons'})
    review_df = review_df.reset_index(drop=True)
    if return_df:
        return review_df
    else:
        for idx in range(len(review_df)):
            print(f"File Name: {review_df.iloc[idx]['File_Name']}") 
            print(f"Reason: {review_df.iloc[idx]['reasons']}")
            print("="*35)


def analyze_quadrant_predictions(log_df, low_conf_threshold=0.6, export_dir=None, split_name=None):

    print(f"Total images processed: {log_df['File_Name'].nunique()}")
    print(f"Total events logged: {len(log_df)}\n")

    # ---- 1. boxes ----
    boxes_df = log_df[log_df['event'] == 'boxes_detected'][['File_Name', 'n_boxes']].drop_duplicates()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    box_counts = boxes_df['n_boxes'].value_counts().sort_index()
    axes[0].bar(box_counts.index.astype(str), box_counts.values, color='steelblue')
    axes[0].set_title('Number of Images by Detected Box Count')
    axes[0].set_xlabel('Boxes Detected')
    axes[0].set_ylabel('Number of Images')
    for i, v in enumerate(box_counts.values):
        axes[0].text(i, v + 0.5, str(v), ha='center')

    correct = (boxes_df['n_boxes'] == 4).sum()
    over = (boxes_df['n_boxes'] > 4).sum()
    under = (boxes_df['n_boxes'] < 4).sum()
    axes[1].pie([correct, over, under],
                labels=[f'Exactly 4\n({correct})', f'More than 4\n({over})', f'Less than 4\n({under})'],
                colors=['#4CAF50', '#FF9800', '#F44336'], autopct='%1.1f%%')
    axes[1].set_title('Detection Accuracy Summary')

    plt.tight_layout()
    if export_dir and split_name:
        plt.savefig(os.path.join(export_dir,split_name,'detected_box_count_summary.png'),dpi=300)
    plt.show()

    # ---- 2. (duplicate for each quad) ----
    dup_df = log_df[log_df['event'] == 'duplicate_quad']
    if len(dup_df) > 0:
        plt.figure(figsize=(8, 5))
        dup_counts = dup_df['quad'].value_counts()
        plt.bar(dup_counts.index, dup_counts.values, color='#FF9800')
        plt.title('Duplicate Predictions per Quadrant')
        plt.ylabel('Count')
        for i, v in enumerate(dup_counts.values):
            plt.text(i, v + 0.3, str(v), ha='center')
        if export_dir and split_name:    
            plt.savefig(os.path.join(export_dir,split_name,'duplicate_predictions_per_quadrant.png'),dpi=300)
        plt.show()
    else:
        print("No duplicate quadrant predictions found.")

    # ---- 3. missing for each quad----
    missing_df = log_df[log_df['event'] == 'missing_quad']
    if len(missing_df) > 0:
        missing_expanded = missing_df['quad'].str.split(' & ').explode()
        plt.figure(figsize=(8, 5))
        missing_counts = missing_expanded.value_counts()
        plt.bar(missing_counts.index, missing_counts.values, color='#F44336')
        plt.title('Missing Predictions per Quadrant')
        plt.ylabel('Count')
        for i, v in enumerate(missing_counts.values):
            plt.text(i, v + 0.3, str(v), ha='center')
        if export_dir and split_name:
            plt.savefig(os.path.join(export_dir,split_name,'missing_predictions_per_quadrant.png'),dpi=300)
        plt.show()
    else:
        print("No missing quadrant predictions found.")

    # ---- 4. Distribution Low Confidence Deapend on the quadrant ----
    low_conf_df = log_df[log_df['event'] == 'low_confidence']
    if len(low_conf_df) > 0:
        plt.figure(figsize=(10, 5))
        for quad in low_conf_df['quad'].unique():
            subset = low_conf_df[low_conf_df['quad'] == quad]
            plt.scatter([quad] * len(subset), subset['confidence'], alpha=0.6)
        plt.axhline(y=low_conf_threshold, color='red', linestyle='--', label=f'Threshold ({low_conf_threshold})')
        plt.title('Low Confidence Predictions by Quadrant')
        plt.ylabel('Confidence')
        plt.legend()
        if export_dir and split_name:
            plt.savefig(os.path.join(export_dir,split_name,'low_confidence_by_quadrant.png'),dpi=300)
        plt.show()

        print(f"\nTotal low confidence warnings: {len(low_conf_df)}")
        print(low_conf_df.groupby('quad')['confidence'].agg(['count', 'mean', 'min']))
    else:
        print("No low confidence predictions found.")

    # ---- 5. Calculate the Avg for high confidence & low confidence ----
    all_conf_events = log_df[log_df['confidence'].notna()]
    if len(all_conf_events) > 0:
        high_conf_events = all_conf_events[all_conf_events['confidence'] >= low_conf_threshold]
        low_conf_events = all_conf_events[all_conf_events['confidence'] < low_conf_threshold]

        high_conf_avg = high_conf_events['confidence'].mean()
        low_conf_avg = low_conf_events['confidence'].mean()
        overall_std = all_conf_events['confidence'].std()

        gap = (high_conf_avg - low_conf_avg) if not np.isnan(low_conf_avg) else None
        low_count = len(low_conf_events)

        print(f"\nAverage HIGH confidence (>= {low_conf_threshold}): {high_conf_avg:.3f}")
        print(f"Average LOW confidence  (< {low_conf_threshold}): {low_conf_avg:.3f}" if not np.isnan(low_conf_avg) else "Average LOW confidence: N/A (no low conf events)")

        if gap is not None:
            # Reliability check: small sample sizes produce misleading gaps
            if low_count < 5:
                print(f"Confidence Gap: {gap:.3f}  (low sample size n={low_count}, result not reliable)")
            else:
                # Classify gap relative to the overall spread of the data (std),
                # instead of using fixed arbitrary thresholds
                if overall_std == 0 or np.isnan(overall_std):
                    severity = 'undetermined (no variance in data)'
                elif gap > 2 * overall_std:
                    severity = 'large, model is inconsistent'
                elif gap > overall_std:
                    severity = 'moderate'
                else:
                    severity = 'small, model is fairly stable'
                print(f"Confidence Gap: {gap:.3f}  (std={overall_std:.3f}, n_low={low_count} -> {severity})")

    # ---- 6. Heatmap: (duplicate confusion) ----
    if len(dup_df) > 0:
        confusion_pairs = []
        for fname in dup_df['File_Name'].unique():
            dup_quads = dup_df[dup_df['File_Name'] == fname]['quad'].tolist()
            missing_quads_for_img = log_df[(log_df['File_Name'] == fname) & (log_df['event'] == 'missing_quad')]
            if len(missing_quads_for_img) > 0:
                missing_list = missing_quads_for_img.iloc[0]['quad'].split(' & ')
                for dq in dup_quads:
                    for mq in missing_list:
                        confusion_pairs.append((dq, mq))

        if confusion_pairs:
            confusion_df = pd.DataFrame(confusion_pairs, columns=['Predicted_Extra', 'Actually_Missing'])
            pivot = confusion_df.groupby(['Predicted_Extra', 'Actually_Missing']).size().unstack(fill_value=0)

            plt.figure(figsize=(8, 6))
            plt.imshow(pivot, cmap='Reds')
            plt.colorbar(label='Count')
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xlabel('Actually Missing Quadrant')
            plt.ylabel('Predicted Extra (Duplicate) Quadrant')
            plt.title('Quadrant Confusion: Duplicate vs Missing (same image)')
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    plt.text(j, i, pivot.iloc[i, j], ha='center', va='center')
            plt.tight_layout()
            if export_dir and split_name:
                plt.savefig(os.path.join(export_dir,split_name,'low_confidence_by_quadrant.png'),dpi=300)
            plt.show()
        else:
            print("\nNo clear duplicate-missing confusion pattern found in the same images.")

    # ---- 7. Worst Images had Misleads ----
    problem_events = log_df[log_df['event'].isin(['duplicate_quad', 'missing_quad', 'low_confidence'])]
    worst_images = problem_events['File_Name'].value_counts().head(10)
    if len(worst_images) > 0:
        print("\nTop 10 problematic images (most warnings):")
        print(worst_images)

    # ---- 8. Export CSV (Optional) ----
    if export_dir and split_name:
        worst_df = problem_events[problem_events['File_Name'].isin(worst_images.index)]
        worst_df.to_csv(os.path.join(export_dir,split_name,f'worst_{split_name}_images.csv'), index=False)
        print(f"\nExported worst images log to: {os.path.join(export_dir,split_name)}")

    return {
        'total_images': boxes_df['File_Name'].nunique(),
        'correct_4_boxes': correct,
        'over_detected': over,
        'under_detected': under,
        'duplicate_events': len(dup_df),
        'missing_events': len(missing_df),
        'low_confidence_events': len(low_conf_df),
        'avg_high_confidence': high_conf_avg if len(all_conf_events) > 0 else None,
        'avg_low_confidence': low_conf_avg if len(all_conf_events) > 0 else None,
    }


def analyze_full_teeth_predictions(log_df, low_conf_threshold=0.6, n_enum_classes=8, export_dir=None, split_name=None):

    print(f"Total images processed: {log_df['File_Name'].nunique()}")
    print(f"Total events logged: {len(log_df)}\n")

    # ---- 1. boxes ----
    # unlike quadrants, a crop can legitimately have anywhere from 0 to
    # n_enum_classes teeth, so there's no single "correct" count to check
    # against. The one thing we CAN say for certain is that more than
    # n_enum_classes boxes in one crop is always wrong.
    boxes_df = log_df[log_df['event'] == 'boxes_detected'][['File_Name', 'n_boxes']].drop_duplicates()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    box_counts = boxes_df['n_boxes'].value_counts().sort_index()
    axes[0].bar(box_counts.index.astype(str), box_counts.values, color='steelblue')
    axes[0].set_title('Number of Images by Raw Detected Box Count')
    axes[0].set_xlabel('Boxes Detected')
    axes[0].set_ylabel('Number of Images')
    for i, v in enumerate(box_counts.values):
        axes[0].text(i, v + 0.5, str(v), ha='center')

    within_range = (boxes_df['n_boxes'] <= n_enum_classes).sum()
    over_range = (boxes_df['n_boxes'] > n_enum_classes).sum()
    axes[1].pie([within_range, over_range],
                labels=[f'<= {n_enum_classes} boxes\n({within_range})', f'> {n_enum_classes} boxes\n({over_range})'],
                colors=['#4CAF50', '#F44336'], autopct='%1.1f%%')
    axes[1].set_title('Raw Box Count vs Max Possible Teeth')

    plt.tight_layout()
    if export_dir and split_name:
        plt.savefig(os.path.join(export_dir,split_name,'detected_box_count_summary.png'),dpi=300)
    plt.show()

    # ---- 2. duplicates (same-location vs different-location) per tooth number ----
    dup_same_df = log_df[log_df['event'] == 'duplicate_same_location']
    dup_diff_df = log_df[log_df['event'] == 'duplicate_diff_location']

    if len(dup_same_df) > 0 or len(dup_diff_df) > 0:
        same_counts = dup_same_df['enum_class'].value_counts()
        diff_counts = dup_diff_df['enum_class'].value_counts()
        all_classes = sorted(set(same_counts.index) | set(diff_counts.index))

        x = np.arange(len(all_classes))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, [same_counts.get(c, 0) for c in all_classes], width, label='Same location', color='#FF9800')
        plt.bar(x + width/2, [diff_counts.get(c, 0) for c in all_classes], width, label='Different location', color='#9C27B0')
        plt.xticks(x, all_classes)
        plt.title('Duplicate Predictions per Tooth Number')
        plt.xlabel('Enumeration Class')
        plt.ylabel('Count')
        plt.legend()
        if export_dir and split_name:
            plt.savefig(os.path.join(export_dir,split_name,'duplicate_predictions_per_tooth.png'),dpi=300)
        plt.show()
    else:
        print("No duplicate tooth predictions found.")

    # ---- 3. missing per tooth number ----
    missing_df = log_df[log_df['event'] == 'missing_teeth']
    if len(missing_df) > 0:
        missing_expanded = missing_df['enum_class'].str.split(' & ').explode()
        plt.figure(figsize=(8, 5))
        missing_counts = missing_expanded.value_counts().sort_index()
        plt.bar(missing_counts.index, missing_counts.values, color='#F44336')
        plt.title('Missing Teeth per Enumeration Class')
        plt.xlabel('Enumeration Class')
        plt.ylabel('Count')
        for i, v in enumerate(missing_counts.values):
            plt.text(i, v + 0.3, str(v), ha='center')
        if export_dir and split_name:
            plt.savefig(os.path.join(export_dir,split_name,'missing_teeth_per_class.png'),dpi=300)
        plt.show()
        print("Note: a missing tooth number isn't necessarily an error, not every quadrant has all", n_enum_classes, "teeth present.")
    else:
        print("No missing teeth found.")

    # ---- 4. already-labeled skips (informational, not a warning) ----
    skipped_df = log_df[log_df['event'] == 'skipped_already_labeled']
    print(f"\nPredictions skipped because the tooth already had a disease label: {len(skipped_df)}")

    # ---- 5. possible cross-quadrant leaks ----
    leak_df = log_df[log_df['event'] == 'possible_cross_quadrant_leak']
    if len(leak_df) > 0:
        plt.figure(figsize=(8, 5))
        leak_counts = leak_df['enum_class'].value_counts().sort_index()
        plt.bar(leak_counts.index, leak_counts.values, color='#795548')
        plt.title('Possible Cross-Quadrant Leaks by Tooth Number')
        plt.xlabel('Enumeration Class')
        plt.ylabel('Count')
        if export_dir and split_name:
            plt.savefig(os.path.join(export_dir,split_name,'possible_leaks_per_class.png'),dpi=300)
        plt.show()
        print(f"Total possible cross-quadrant leaks flagged: {len(leak_df)} (heuristic, worth a manual look)")
    else:
        print("No possible cross-quadrant leaks flagged.")

    # ---- 6. possible background predictions ----
    bg_df = log_df[log_df['event'] == 'possible_background_prediction']
    if len(bg_df) > 0:
        plt.figure(figsize=(8, 5))
        bg_counts = bg_df['enum_class'].value_counts().sort_index()
        plt.bar(bg_counts.index, bg_counts.values, color='#607D8B')
        plt.title('Possible Background Predictions by Tooth Number')
        plt.xlabel('Enumeration Class')
        plt.ylabel('Count')
        if export_dir and split_name:
            plt.savefig(os.path.join(export_dir,split_name,'possible_background_per_class.png'),dpi=300)
        plt.show()
        print(f"Total possible background predictions flagged: {len(bg_df)} (heuristic, worth a manual look)")
    else:
        print("No possible background predictions flagged.")

    # ---- 7. Distribution of low confidence, depending on tooth number ----
    low_conf_df = log_df[log_df['event'] == 'low_confidence']
    if len(low_conf_df) > 0:
        plt.figure(figsize=(10, 5))
        for cls_id in sorted(low_conf_df['enum_class'].unique()):
            subset = low_conf_df[low_conf_df['enum_class'] == cls_id]
            plt.scatter([cls_id] * len(subset), subset['confidence'], alpha=0.6)
        plt.axhline(y=low_conf_threshold, color='red', linestyle='--', label=f'Threshold ({low_conf_threshold})')
        plt.title('Low Confidence Predictions by Tooth Number')
        plt.xlabel('Enumeration Class')
        plt.ylabel('Confidence')
        plt.legend()
        if export_dir and split_name:
            plt.savefig(os.path.join(export_dir,split_name,'low_confidence_by_tooth.png'),dpi=300)
        plt.show()

        print(f"\nTotal low confidence warnings: {len(low_conf_df)}")
        print(low_conf_df.groupby('enum_class')['confidence'].agg(['count', 'mean', 'min']))
    else:
        print("No low confidence predictions found.")

    # ---- 8. Calculate the avg for high confidence & low confidence ----
    all_conf_events = log_df[log_df['confidence'].notna()]
    high_conf_avg = low_conf_avg = None
    if len(all_conf_events) > 0:
        high_conf_events = all_conf_events[all_conf_events['confidence'] >= low_conf_threshold]
        low_conf_events = all_conf_events[all_conf_events['confidence'] < low_conf_threshold]

        high_conf_avg = high_conf_events['confidence'].mean()
        low_conf_avg = low_conf_events['confidence'].mean()
        overall_std = all_conf_events['confidence'].std()

        gap = (high_conf_avg - low_conf_avg) if not np.isnan(low_conf_avg) else None
        low_count = len(low_conf_events)

        print(f"\nAverage HIGH confidence (>= {low_conf_threshold}): {high_conf_avg:.3f}")
        print(f"Average LOW confidence  (< {low_conf_threshold}): {low_conf_avg:.3f}" if not np.isnan(low_conf_avg) else "Average LOW confidence: N/A (no low conf events)")

        if gap is not None:
            # Reliability check: small sample sizes produce misleading gaps
            if low_count < 5:
                print(f"Confidence Gap: {gap:.3f}  (low sample size n={low_count}, result not reliable)")
            else:
                # Classify gap relative to the overall spread of the data (std),
                # instead of using fixed arbitrary thresholds
                if overall_std == 0 or np.isnan(overall_std):
                    severity = 'undetermined (no variance in data)'
                elif gap > 2 * overall_std:
                    severity = 'large, model is inconsistent'
                elif gap > overall_std:
                    severity = 'moderate'
                else:
                    severity = 'small, model is fairly stable'
                print(f"Confidence Gap: {gap:.3f}  (std={overall_std:.3f}, n_low={low_count} -> {severity})")

    # ---- 9. Heatmap: confusion between a duplicated tooth and a missing one ----
    # only duplicate_diff_location is used here: two boxes for the same tooth
    # number in different spots often means the model actually confused that
    # tooth number with a genuinely different, missing one in the same image
    if len(dup_diff_df) > 0:
        confusion_pairs = []
        for fname in dup_diff_df['File_Name'].unique():
            dup_classes = dup_diff_df[dup_diff_df['File_Name'] == fname]['enum_class'].tolist()
            missing_for_img = log_df[(log_df['File_Name'] == fname) & (log_df['event'] == 'missing_teeth')]
            if len(missing_for_img) > 0:
                missing_list = missing_for_img.iloc[0]['enum_class'].split(' & ')
                for dc in dup_classes:
                    for mc in missing_list:
                        confusion_pairs.append((dc, mc))

        if confusion_pairs:
            confusion_df = pd.DataFrame(confusion_pairs, columns=['Predicted_Extra', 'Actually_Missing'])
            pivot = confusion_df.groupby(['Predicted_Extra', 'Actually_Missing']).size().unstack(fill_value=0)

            plt.figure(figsize=(8, 6))
            plt.imshow(pivot, cmap='Reds')
            plt.colorbar(label='Count')
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xlabel('Actually Missing Tooth')
            plt.ylabel('Predicted Extra (Duplicate) Tooth')
            plt.title('Tooth Confusion: Duplicate vs Missing (same image)')
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    plt.text(j, i, pivot.iloc[i, j], ha='center', va='center')
            plt.tight_layout()
            if export_dir and split_name:
                plt.savefig(os.path.join(export_dir,split_name,'tooth_confusion_heatmap.png'),dpi=300)
            plt.show()
        else:
            print("\nNo clear duplicate-missing confusion pattern found in the same images.")

    # ---- 10. Worst images had misleads ----
    problem_events = log_df[log_df['event'].isin([
        'duplicate_same_location', 'duplicate_diff_location',
        'possible_cross_quadrant_leak', 'possible_background_prediction',
        'low_confidence', 'missing_teeth'
    ])]
    worst_images = problem_events['File_Name'].value_counts().head(10)
    if len(worst_images) > 0:
        print("\nTop 10 problematic images (most warnings):")
        print(worst_images)

    # ---- 11. Export CSV (Optional) ----
    if export_dir and split_name:
        worst_df = problem_events[problem_events['File_Name'].isin(worst_images.index)]
        worst_df.to_csv(os.path.join(export_dir,split_name,f'worst_{split_name}_images.csv'), index=False)
        print(f"\nExported worst images log to: {os.path.join(export_dir,split_name)}")

    return {
        'total_images': boxes_df['File_Name'].nunique(),
        'within_max_teeth': within_range,
        'over_max_teeth': over_range,
        'duplicate_same_location_events': len(dup_same_df),
        'duplicate_diff_location_events': len(dup_diff_df),
        'possible_leak_events': len(leak_df),
        'possible_background_events': len(bg_df),
        'skipped_already_labeled_events': len(skipped_df),
        'missing_events': len(missing_df),
        'low_confidence_events': len(low_conf_df),
        'avg_high_confidence': high_conf_avg,
        'avg_low_confidence': low_conf_avg,
    }


def compare_best_vs_last(data_yaml,models_yaml=None, stage='quadrant', split_name='test',
                          original_images_path=None, annotations_df_path=None,
                          results_csv_path=None, main_result_path=None,
                          conf_threshold=0.3, verbose=False):


    images_root = data_yaml['val' if split_name == 'valid' else split_name] 
    labels_root = data_yaml['val' if split_name == 'valid' else split_name].replace('images','labels') 

    if stage.strip().lower() not in ('quadrant', 'enumeration', 'enumeration_continued'):
        print(f"Unknown stage '{stage}', expected 'quadrant', 'enumeration' or 'enumeration_continued'.")
        return

    result = {}
    if results_csv_path:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()

        if "epoch" not in df.columns:
            print("Column 'epoch' not found in the file.")
            return

        map50_col = [c for c in df.columns if "mAP50" in c and "50-95" not in c]
        map50_95_col = [c for c in df.columns if "mAP50-95" in c]

        if not map50_col or not map50_95_col:
            print("Could not find mAP50 or mAP50-95 columns in the file.")
            return

        m50 = map50_col[0]
        m50_95 = map50_95_col[0]

        best_idx = df[m50_95].idxmax()
        best_row = df.loc[best_idx]
        best_epoch = int(best_row["epoch"])

        last_row = df.iloc[-1]
        last_epoch = int(last_row["epoch"])

        best_map50 = best_row[m50]
        best_map50_95 = best_row[m50_95]

        last_map50 = last_row[m50]
        last_map50_95 = last_row[m50_95]

        print("Comparison: Best Epoch vs Last Epoch")
        print(f"Best Epoch ({best_epoch + 1}):")
        print(f"  mAP50: {best_map50:.4f}")
        print(f"  mAP50-95: {best_map50_95:.4f}")

        print(f"Last Epoch ({last_epoch + 1}):")
        print(f"  mAP50: {last_map50:.4f}")
        print(f"  mAP50-95: {last_map50_95:.4f}")

        diff_map50_95 = best_map50_95 - last_map50_95

        print("Verdict (based on training metrics only):")
        if best_epoch == last_epoch:
            print("Best and Last are the same epoch, so the last epoch is also the best one.")
        elif diff_map50_95 > 0.001:
            print(f"Best Epoch ({best_epoch + 1}) is better than Last Epoch ({last_epoch + 1}).")
            print(f"Difference in mAP50-95: +{diff_map50_95:.4f}")
            print("This gap can indicate some overfitting toward the end of training.")
        else:
            print("The difference is very small, meaning the model was fairly stable by the end.")

        result = {
            "best_epoch": best_epoch + 1,
            "last_epoch": last_epoch + 1,
            "best_map50": best_map50,
            "best_map50_95": best_map50_95,
            "last_map50": last_map50,
            "last_map50_95": last_map50_95,
        }

        if models_yaml is None:
            return result

    # ---- check we have what this stage needs before touching the test set ----
    annotations_df = None
    if stage.strip().lower() == 'quadrant':
        if original_images_path is None or annotations_df_path is None:
            print("stage='quad' needs original_images_path and annotations_df_path, skipping test set check.")
            return result
        annotations_df = pd.read_pickle(annotations_df_path)
    elif stage.strip().lower() == 'enumeration' or stage.strip().lower() == 'enumeration_continued':
        if images_root is None or labels_root is None:
            print("stage='enumeration' and 'enumeration_continued' needs images_root and labels_root, skipping test set check.")
            return result

    for model_key in models_yaml.keys():
        if stage.strip().lower() in model_key:
            model_name = model_key

    models_config = models_yaml[model_name]

    print("")
    print(f"Running best and last weights on the test set (stage: {stage})")

    weight_results = {}

    for checkpoint_name, weights_path in models_config.items():

        print(f"{model_name} - {checkpoint_name}")
        model = YOLO(weights_path)

        if stage.strip().lower() == 'quadrant':
            log_df = export_quadrants_using_quad_model(
                model,
                original_images_path,
                annotations_df,
                export_labels=False,
                export_images=False,
                conf_threshold=conf_threshold,
                clear_existing=False,
                verbose=verbose,
            )
            n_duplicate = (log_df['event'] == 'duplicate_quad').sum()
            n_missing = (log_df['event'] == 'missing_quad').sum()
            n_leak = 0
            n_background = 0

            # ---- 7. Worst Images had Misleads ----
            problem_events = log_df[log_df['event'].isin(['duplicate_quad', 'missing_quad', 'low_confidence'])]
            worst_images = problem_events['File_Name'].value_counts().head(10)
            if len(worst_images) > 0:
                print("\nTop 10 problematic images (most warnings):")
                worst_df = problem_events[problem_events['File_Name'].isin(worst_images.index)]


        else:  # stage == 'enum' or 'enum_cont'
            log_df = export_teeth_in_quad_using_enum_model(
                model,
                images_root,
                labels_root,
                export_labels=False,
                export_images=False,
                conf_threshold=conf_threshold,
                clear_existing=False,
                verbose=verbose,
            )
            # same_location + diff_location both represent a dropped duplicate box,
            # so they're combined into one duplicate count here for comparability
            # with the quad stage
            n_duplicate = log_df['event'].isin(['duplicate_same_location', 'duplicate_diff_location']).sum()
            n_missing = (log_df['event'] == 'missing_teeth').sum()
            n_leak = (log_df['event'] == 'possible_cross_quadrant_leak').sum()
            n_background = (log_df['event'] == 'possible_background_prediction').sum()

            # ---- Worst images had misleads ----
            problem_events = log_df[log_df['event'].isin([
                'duplicate_same_location', 'duplicate_diff_location',
                'possible_cross_quadrant_leak', 'possible_background_prediction',
                'low_confidence', 'missing_teeth'
            ])]
            worst_images = problem_events['File_Name'].value_counts().head(10)
            if len(worst_images) > 0:
                print("\nTop 10 problematic images (most warnings):")
                worst_df = problem_events[problem_events['File_Name'].isin(worst_images.index)]

        n_low_conf = (log_df['event'] == 'low_confidence').sum()
        n_success = (log_df['event'] == 'successful_detection').sum()

        avg_conf = log_df.loc[log_df['event'] == 'successful_detection', 'confidence'].mean()

        total_errors = n_duplicate + n_low_conf + n_leak + n_background

        weight_results[f"{model_name}_{checkpoint_name}"] = {
            'model_name': model_name,
            'tag': checkpoint_name,
            'n_duplicate': n_duplicate,
            'n_low_conf': n_low_conf,
            'n_missing': n_missing,
            'n_leak': n_leak,
            'n_background': n_background,
            'n_success': n_success,
            'avg_confidence': avg_conf,
            'total_errors': total_errors,
            'worst_images_df': worst_df,
            'log_df': log_df,
        }

        if stage.strip().lower() == 'enumeration' or stage.strip().lower() == 'enumeration_continued':
            print(f"  duplicates: {n_duplicate}, low_confidence: {n_low_conf}, missing: {n_missing}, "
                  f"leaks: {n_leak}, background: {n_background}, successful: {n_success}, avg_conf: {avg_conf:.4f}")
        else:
            print(f"  duplicates: {n_duplicate}, low_confidence: {n_low_conf}, missing: {n_missing}, "
                  f"successful: {n_success}, avg_conf: {avg_conf:.4f}")

    print("")
    print("Test set summary")
    for key, r in weight_results.items():
        print(f"{key}: total_errors={r['total_errors']}, avg_confidence={r['avg_confidence']:.4f}")

    lowest_error_key = min(weight_results, key=lambda k: weight_results[k]['total_errors'])
    print("")
    print(f"Lowest total errors on test set: {lowest_error_key} with {weight_results[lowest_error_key]['total_errors']} errors")

    result['test_set_results'] = weight_results

    if main_result_path and split_name:
        export_result_path = os.path.join(main_result_path,split_name)
        import pickle
        file_path = os.path.join(export_result_path, 'result_summary.pkl')
        with open(file_path, "wb") as f:
            pickle.dump(result, f)

    return result



# ==========================================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# ==========================================================================================================================================================

 
 
def balance_with_oversampling(originals_root, random_state=42, target_class=None, target_count=None):
    """
    Balances class folders under originals_root.

    Args:
        originals_root: folder containing one subfolder per class, original
            (non-augmented) images only
        random_state: for reproducibility
        target_class: optional class name to oversample. If provided, only this
            class folder will be processed.
        target_count: optional target number of images. If None, the function
            uses the largest class count found under originals_root.

    Returns:
        dict of class_name -> number of copies created
    """
    random.seed(random_state)
    originals_root = Path(originals_root)

    class_dirs = {d.name: d for d in originals_root.iterdir() if d.is_dir()}
    if not class_dirs:
        raise ValueError(f"No class folders found in {originals_root}")

    counts = {name: len([p for p in d.iterdir() if p.is_file()]) for name, d in class_dirs.items()}
    if target_count is None:
        target_count = max(counts.values())

    if target_class is not None:
        if target_class not in class_dirs:
            raise ValueError(f"Class '{target_class}' not found in {originals_root}")
        target_classes = [target_class]
    else:
        target_classes = list(class_dirs.keys())

    print("Current counts:", counts)
    print("Target per class:", target_count)

    created = {}
    for name in target_classes:
        class_dir = class_dirs[name]
        images = [p for p in class_dir.iterdir() if p.is_file()]
        needed = target_count - len(images)
        created[name] = needed

        if needed <= 0:
            continue

        pool = images.copy()
        random.shuffle(pool)
        copy_i = 0
        while needed > 0:
            src = pool[copy_i % len(pool)]
            copy_i += 1
            dest = class_dir / f"{src.stem}_copy{copy_i}{src.suffix}"
            shutil.copy2(src, dest)
            needed -= 1

    print("Copies created per class:", created)
    return created
 
 

 
class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification. Down-weights easy, already-correct
    predictions so the model focuses more on hard or minority-class examples,
    useful here as an alternative to oversampling for class imbalance.

    Args:
        alpha: class weights (tensor of shape [num_classes]), or None for no weighting.
               Give rare classes a higher value if you want extra emphasis on them.
        gamma: focusing parameter. Higher gamma pushes the model harder toward hard examples.
               2.0 is the common default.
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = exp(-ce_loss)  # probability the model assigned to the correct class
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def denorm(imgs):
    from torch import tensor
    """
    Reverse ImageNet normalization so images can be displayed correctly.

    Args:
        imgs: normalized tensor of shape (B, C, H, W) or (C, H, W)

    Returns:
        tensor in [0, 1] range with same shape as input
    """
    if imgs.dim() == 3:
        mean = tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(imgs.device)
        std = tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(imgs.device)
        return imgs * std + mean

    if imgs.dim() == 4:
        mean = tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs.device)
        std = tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(imgs.device)
        return imgs * std + mean

    raise ValueError(f"denorm expects 3D or 4D tensor, got shape {tuple(imgs.shape)}")

def get_transforms(image_size, apply_on_train=False):
    from torchvision.transforms import v2 ; from torch import float32

    """
    Build a transform pipeline for training or validation/test.

    Args:
        size: target square image size in pixels
        apply_on_train: if True, adds augmentation steps before normalization

    Returns:
        v2.Compose transform pipeline
    """
    base = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(float32, scale=True),
    ]

    augmentation = [
        v2.RandomAutocontrast(p=0.5),
        v2.RandomEqualize(p=0.5),
    ]

    # ImageNet mean/std normalization (required for pretrained models)
    tail = [v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    if apply_on_train:
        return v2.Compose(base + augmentation + tail)
    return v2.Compose(base + tail)


def build_swin_model(
    num_classes,
    lr=1e-3,
    weight_decay=0.0001,
    epochs=10,
    freeze_backbone=True,
    unfreeze_layers=None,
    scheduler_type='cosine',
    dropout=0.0,
    steps_per_epoch=None,):
    """
    Build a simple Swin Transformer classifier with optional freezing, layer unfreezing, dropout, and scheduler.
    If you want to unfreeze specific layers by index, use names like:
    unfreeze_layers=['features.0', 'features.1', 'norm'] (Swin Small Model have 7 featuers layers)
    This will unfreeze any parameter names containing those strings.
    """
    from torchvision.models import swin_v2_t,Swin_V2_T_Weights
    from torch.optim import AdamW,lr_scheduler
    from torch import nn

    model = swin_v2_t(Swin_V2_T_Weights.DEFAULT)

    in_features = model.head.in_features
    if dropout > 0:
        model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.head = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    if unfreeze_layers:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type in {'cosine', 'cosineannealing'}:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    elif scheduler_type in {'step', 'steplr'}:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif scheduler_type in {'reduce_on_plateau', 'reduceLROnPlateau'}:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    elif scheduler_type in {'one_cycle', 'on_cycle'}:
        if steps_per_epoch is None:
            steps_per_epoch = 1
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )
    else:
        scheduler = None

    return model, optimizer, scheduler


def predict_classifier(model, dataloader, device, class_names,disease_class_idx=None, threshold=None,
             show_plot=False, num_samples=20, save_plot_path=os.getcwd(), figsize=(12, 10), return_probs=False):
    """
    Runs the model over a dataloader and returns true/predicted labels.

    Args:
        model, dataloader, device, class_names: as before
        disease_class_idx: index of the "disease" class in class_names,
                            required if threshold is set
        threshold: if set, classifies as disease_class_idx whenever its
                   probability exceeds this value, instead of using argmax.
                   Lower than 0.5 favors catching more disease cases.

        return_probs: if True, also returns a list of per-image class-probability
                      dicts (all_probs), needed for confidence charts downstream.
                   
    Returns:
        all_labels, all_preds
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if threshold is not None:
                disease_prob = probs[:, disease_class_idx]
                preds = torch.where(disease_prob > threshold,
                                     torch.tensor(disease_class_idx, device=device),
                                     torch.tensor(1 - disease_class_idx, device=device))
            else:
                _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            if return_probs:
                all_probs.extend(probs.cpu().numpy().tolist())

    if show_plot:
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None:
            raise ValueError("dataloader must provide a dataset attribute for plotting")

        if class_names is None:
            class_names = getattr(dataset, "classes", None)
        if class_names is None:
            class_names = [str(i) for i in range(max(len(np.unique(all_labels)), 2))]

        sample_size = min(num_samples, len(dataset))
        sample_indices = np.random.choice(len(dataset), size=sample_size, replace=False)

        n_cols = 5
        n_rows = int(np.ceil(sample_size / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for ax, idx in zip(axes, sample_indices):
            img, true_label = dataset[idx]
            img = denorm(img).permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)

            true_idx = int(all_labels[idx])
            pred_idx = int(all_preds[idx])
            correct = pred_idx == true_idx

            ax.imshow(img)
            ax.set_title(
                f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]}",
                color="green" if correct else "red",
                fontsize=9,
            )
            ax.axis("off")

        for ax in axes[sample_size:]:
            ax.axis("off")

        plt.suptitle("Test Set Predictions", fontsize=14, y=1.01)
        plt.tight_layout()

        plt.savefig(os.path.join(save_plot_path,'test_set_predicted.png'), dpi=300, bbox_inches='tight')

        plt.show()

    if return_probs:
        return all_labels, all_preds, all_probs
    return all_labels, all_preds

