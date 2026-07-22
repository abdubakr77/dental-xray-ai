import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from src.vis import visualize_augmentation
from tqdm import tqdm

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



def export_enum_by_quad_using_model(quadrant_model, enum_images_path, enum_df,
                                    output_root=os.getcwd(),
                                    conf_threshold=0.3,
                                    debugging=False, debug_limit=5,
                                    clear_existing=True):

    # ---- check + clear existing images/labels ----
    if clear_existing and not debugging:
        imgs_path = os.path.join(output_root, 'images')
        labels_path = os.path.join(output_root, 'labels')

        any_existing = (os.path.exists(imgs_path) and len(os.listdir(imgs_path)) > 0) or \
                        (os.path.exists(labels_path) and len(os.listdir(labels_path)) > 0)

        if any_existing:
            print("Warning: Found existing images/labels in the output directory.")
            confirm = input("Do you want to delete them all before re-exporting? - (y or n): ").lower().strip()

            if confirm == 'y':
                deleted_count = 0
                failed_count = 0
                for sub_path in [imgs_path, labels_path]:
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

    for fname in tqdm(enum_df['File_Name'].unique(), desc='Cropping using trained quadrant model'):

        if debugging:
            fname = np.random.choice(enum_df['File_Name'].unique())

        img_path = os.path.join(enum_images_path, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = quadrant_model.predict(img_path, conf=conf_threshold, verbose=False)[0]
        n_predicted_boxes = len(results.boxes)

        for n,box in enumerate(results.boxes,start=1):

            quad_class_id = int(box.cls[0])
            quad_name = quadrant_model.names[quad_class_id]
            confidence = np.round(float(box.conf), 2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            crop_h, crop_w = cropped_img.shape[:2]
            if crop_h == 0 or crop_w == 0:
                continue

            teeth_rows = enum_df[(enum_df['File_Name'] == fname) & (enum_df['Quad'] == quad_name)]

            new_labels = []
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

                if n > 4 and n == n_predicted_boxes:
                    print(f'Warning: Model Detected {n_predicted_boxes} Boxes And Predicted {quad_name} Again With {confidence} Confidence In This Image Name: {fname.split('.')[0]}')
                    if os.path.exists(img_out_path):
                        print("Skipped Successfuly!")
                        continue
                elif n < 4 and n == n_predicted_boxes:
                    print(f'Warning: Model Detected {n_predicted_boxes} Boxes And Can\'t Predict {quadrant_model.names[n]} In This Image Name: {fname.split('.')[0]}')
                    
                if confidence < 0.6:
                    print(f"Warning: Low Confidence Alert! Got {confidence} At Quadrant {quad_name} In Image Name: {fname.split('.')[0]}")
                
                if os.path.exists(img_out_path):
                    raise FileExistsError(
                        f"Error: File already exists at {img_out_path}.\n"
                        f"Set clear_existing=True and re-run, or delete manually first."
                    )

                cv2.imwrite(img_out_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                with open(label_out_path, 'w') as f:
                    for cls_id, cx, cy, nw, nh in new_labels:
                        f.write(f"{cls_id} {cx} {cy} {nw} {nh}\n")

        if debugging and debug_count >= debug_limit:
            visualize_augmentation(all_images, all_bboxes, all_labels, titles=all_filenames)
            break