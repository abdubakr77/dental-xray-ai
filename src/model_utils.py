import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from src.vis import visualize_augmentation
from tqdm import tqdm
import pandas as pd
import yaml
from ultralytics import YOLO

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


def export_enum_by_quad_using_model(quadrant_model, original_images_path, annotations_df=None,
                                    output_root=os.getcwd(),
                                    conf_threshold=0.3,
                                    debugging=False, debug_limit=5,
                                    clear_existing=True,
                                    verbose=False,
                                    export_labels=True,
                                    export_images=True):

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

    for fname in tqdm(file_list, desc='Cropping using trained quadrant model'):

        if debugging:
            fname = np.random.choice(file_list)

        img_path = os.path.join(original_images_path, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = quadrant_model.predict(img_path, conf=conf_threshold, verbose=False)[0]
        n_predicted_boxes = len(results.boxes)
        expected_quads = ["Upper Right", "Upper Left", "Lower Left", "Lower Right"]

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
                    if os.path.exists(img_out_path):
                        if verbose:
                            print("Skipped Successfuly!")
                        continue

                expected_quads.remove(quad_name)

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
            'File_Name': fname, 'event': 'boxes_detected',
            'quad': None, 'confidence': None,
            'n_boxes': n_predicted_boxes,
            'crop_area': crop_h * crop_w
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

    log_df = pd.DataFrame(log_records)
    return log_df


def analyze_quadrant_predictions(log_df, low_conf_threshold=0.6, export_worst_csv=None):

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
        plt.show()

        print(f"\nTotal low confidence warnings: {len(low_conf_df)}")
        print(low_conf_df.groupby('quad')['confidence'].agg(['count', 'mean', 'min']))
    else:
        print("No low confidence predictions found.")

    # ---- 5. Calculate the Avg for high confidence & low confidence ----
    all_conf_events = log_df[log_df['confidence'].notna()]
    if len(all_conf_events) > 0:
        high_conf_avg = all_conf_events[all_conf_events['confidence'] >= low_conf_threshold]['confidence'].mean()
        low_conf_avg = all_conf_events[all_conf_events['confidence'] < low_conf_threshold]['confidence'].mean()
        gap = (high_conf_avg - low_conf_avg) if not np.isnan(low_conf_avg) else None

        print(f"\nAverage HIGH confidence (>= {low_conf_threshold}): {high_conf_avg:.3f}")
        print(f"Average LOW confidence  (< {low_conf_threshold}): {low_conf_avg:.3f}" if not np.isnan(low_conf_avg) else "Average LOW confidence: N/A (no low conf events)")
        if gap is not None:
            print(f"Confidence Gap: {gap:.3f}  ({'large, model is inconsistent' if gap > 0.4 else 'moderate' if gap > 0.2 else 'small, model is fairly stable'})")

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
    if export_worst_csv:
        worst_df = problem_events[problem_events['File_Name'].isin(worst_images.index)]
        worst_df.to_csv(export_worst_csv, index=False)
        print(f"\nExported worst images log to: {export_worst_csv}")

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


def compare_best_vs_last(results_csv_path, models_yaml=None, model_name=None, original_images_path=None,
                          test_df_path=None, conf_threshold=0.3, verbose=False):

    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()
    test_df = pd.read_pickle(test_df_path)

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

    if original_images_path is None or test_df is None:
        print("models_yaml_path was given but enum_images_path or annotations_df is missing, skipping test set check.")
        return result

    for model_key in models_yaml.keys():
        if model_name in model_key:
            model_name = model_key
    
    models_config = models_yaml[model_name]

    print("")
    print("Running best and last weights on the test set")

    weight_results = {}

    for checkpoint_name, weights_path in models_config.items():

        print(f"{model_name} - {checkpoint_name}")
        model = YOLO(weights_path)

        log_df = export_enum_by_quad_using_model(
            model,
            original_images_path,
            test_df,
            export_labels=False,
            export_images=False,
            conf_threshold=conf_threshold,
            clear_existing=False,
            verbose=verbose,
        )

        n_duplicate = (log_df['event'] == 'duplicate_quad').sum()
        n_low_conf = (log_df['event'] == 'low_confidence').sum()
        n_missing = (log_df['event'] == 'missing_quad').sum()
        n_success = (log_df['event'] == 'successful_detection').sum()

        avg_conf = log_df.loc[log_df['event'] == 'successful_detection', 'confidence'].mean()

        total_errors = n_duplicate + n_low_conf + n_missing

        weight_results[f"{model_name}_{checkpoint_name}"] = {
            'model_name': model_name,
            'tag': checkpoint_name,
            'n_duplicate': n_duplicate,
            'n_low_conf': n_low_conf,
            'n_missing': n_missing,
            'n_success': n_success,
            'avg_confidence': avg_conf,
            'total_errors': total_errors,
            'log_df': log_df,
        }

        print(f"  duplicates: {n_duplicate}, low_confidence: {n_low_conf}, missing: {n_missing}, successful: {n_success}, avg_conf: {avg_conf:.4f}")

    print("")
    print("Test set summary")
    for key, r in weight_results.items():
        print(f"{key}: total_errors={r['total_errors']}, avg_confidence={r['avg_confidence']:.4f}")

    lowest_error_key = min(weight_results, key=lambda k: weight_results[k]['total_errors'])
    print("")
    print(f"Lowest total errors on test set: {lowest_error_key} with {weight_results[lowest_error_key]['total_errors']} errors")

    result['test_set_results'] = weight_results
    return result