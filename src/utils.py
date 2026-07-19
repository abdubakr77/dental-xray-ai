import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import os
import shutil
from time import sleep
from tqdm import tqdm
import cv2
import sys
sys.path.append(os.path.abspath('..'))

from src.vis import visualize_augmentation
import albumentations as A
from IPython.display import clear_output
import numpy as np 

# pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def compute_iou(box1, box2):
    # Must boxes format are: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    return intersection / union

def find_duplicate_boxes(df, iou_threshold=0.9):
    duplicates = []
    
    for fname in df['File_Name'].unique():
        rows = df[df['File_Name'] == fname].reset_index()
        n = len(rows)
        
        for i in range(n):
            for j in range(i+1, n):
                box1 = rows.iloc[i]['Bbox']
                box2 = rows.iloc[j]['Bbox']
                iou = compute_iou(box1, box2)
                diseases_text = f'{rows.iloc[i]['Disease_Name']} {rows.iloc[j]['Disease_Name']}'
                sorted_diseases_text = sorted(diseases_text.split())
                if iou >= iou_threshold:
                    duplicates.append({
                        'fname': fname,
                        'index_1': rows.iloc[i]['index'],
                        'index_2': rows.iloc[j]['index'],
                        'iou': iou,
                        'diseases': f'{sorted_diseases_text[0]} & {sorted_diseases_text[1]}',
                    })

    print(f"Number of duplicate Boxes: {len(duplicates)}")
    
    return pd.DataFrame(duplicates)




def multilabel_train_val_test_split(df,y,test_size=0.2,apply_check = False):


    grouped = (
        df.groupby("File_Name")[y]
        .apply(list)
        .reset_index()
    )


    mlb = MultiLabelBinarizer()

    Y = mlb.fit_transform(grouped[y])


    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42
    )

    train_idx, temp_idx = next(msss.split(grouped["File_Name"], Y))

    train_images = grouped.iloc[train_idx]
    temp_images = grouped.iloc[temp_idx]



    Y_temp = Y[temp_idx]

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=42
    )

    val_idx, test_idx = next(
        msss2.split(temp_images["File_Name"], Y_temp)
    )

    val_images = temp_images.iloc[val_idx]
    test_images = temp_images.iloc[test_idx]



    train_df = df[df["File_Name"].isin(train_images["File_Name"])].reset_index(drop=True)

    val_df = df[df["File_Name"].isin(val_images["File_Name"])].reset_index(drop=True)

    test_df = df[df["File_Name"].isin(test_images["File_Name"])].reset_index(drop=True)

    if apply_check:
        print()

        print("Train Images :", train_images.shape[0])
        print("Val Images   :", val_images.shape[0])
        print("Test Images  :", test_images.shape[0])

        print()

        print("Train Rows :", len(train_df))
        print("Val Rows   :", len(val_df))
        print("Test Rows  :", len(test_df))

        print()

        train_files = set(train_df["File_Name"])
        val_files = set(val_df["File_Name"])
        test_files = set(test_df["File_Name"])

        assert train_files.isdisjoint(val_files)
        assert train_files.isdisjoint(test_files)
        assert val_files.isdisjoint(test_files)

        print("Perfect! No Data Leakage Found.")

    return train_df, val_df , test_df



def convert_to_yolo(target_col:str, images_path:str ,output_root: str, train_df, valid_df, test_df=None):
    print("WARNING: Please Check all txt files in labels folder are cleared because this function is recommended to run it only once.\nYou Have 5 seconds from now if you need to stop the code!")
    sleep(5)
    # clear_output(wait=True)

    ds_partitions = {'train_df':train_df,
                     'valid_df':valid_df,
                     'test_df':test_df,}
    

    for name,df in ds_partitions.items():
        for fname in tqdm(df['File_Name'].unique().tolist(),f'{name} Is Processing Now...'):

            filtered_df = df[df['File_Name'] == fname]

            output_img_name_no_ext = f"{fname.split('.')[0]}"

            for idx in range(len(filtered_df)):

                x, y, w, h = filtered_df.iloc[idx]['Bbox']
                img_h = filtered_df.iloc[idx]['Height']
                img_w = filtered_df.iloc[idx]['Width']

                x1 = (x + w / 2) / img_w
                y1 = (y + h / 2) / img_h
                x2 = w / img_w
                y2 = h / img_h

                # x_back = (x1 - x2 / 2) * img_w
                # y_back = (y1 - y2 / 2) * img_h
                # w_back = x2 * img_w
                # h_back = y2 * img_h

                # print("Normalized:", x1, y1, x2, y2)
                # print("Back to pixels:", x_back, y_back, w_back, h_back)
                # print("Original was:  ", x, y, w, h)

                if target_col == 'Disease_Name':
                    cls_id = ["impacted", "caries", "periapical","deep_caries"].index(filtered_df.iloc[idx][target_col])
                elif target_col == 'Enumeration':
                    cls_id = 0
                    if 'Disease_Name' in df.columns:
                        output_img_name_no_ext = f'{fname.split('.')[0]}_dis_teeth'
                else:
                    cls_id = ["Upper Right", "Upper Left","Lower Left","Lower Right"].index(filtered_df.iloc[idx][target_col])

                if   'train' in name: labels_path = os.path.join(output_root,'train','labels',output_img_name_no_ext+'.txt'); imgs_path = os.path.join(output_root,'train','images',output_img_name_no_ext+'.png')
                elif 'valid' in name: labels_path = os.path.join(output_root,'valid','labels',output_img_name_no_ext+'.txt'); imgs_path = os.path.join(output_root,'valid','images',output_img_name_no_ext+'.png')
                else                : labels_path = os.path.join(output_root,'test','labels',output_img_name_no_ext+'.txt'); imgs_path = os.path.join(output_root,'test','images',output_img_name_no_ext+'.png')

                with open(labels_path,'a') as f:
                    f.write(f'{cls_id}  {x1}  {y1}  {x2}  {y2}\n')

            if os.path.exists(imgs_path):
                raise FileExistsError(f"Error: There is a files are existed...\nIf you need to ignore it then type (None) in {name} parameter to not duplicate the files!\nOr make sure that you deleted the files")
            
            shutil.copy2(os.path.join(images_path,fname),imgs_path)
                # break 

        print("Done!")


def crop_image(image,x,y,w,h):
    img_h, img_w = image.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    return image[y1:y2, x1:x2]



def prepare_disease_classifier(images_path,output_root,train_df,valid_df,test_df=None):

    ds_partitions = {'train_df':train_df,
                     'valid_df':valid_df,
                     'test_df':test_df,}
    

    for name,df in ds_partitions.items():
        for fname in tqdm(df['File_Name'].unique().tolist(),f'{name} Is Processing Now...'):

            filtered_df = df[df['File_Name'] == fname]

            fname_no_ext = f"{fname.split('.')[0]}"

            for idx in range(len(filtered_df)):

                x, y, w, h = filtered_df.iloc[idx]['Bbox']

                train_path = os.path.join(output_root,'train')
                valid_path = os.path.join(output_root,'valid')
                test_path = os.path.join(output_root,'test')

                if 'Disease_Name' in df.columns:
                    dis_list = ["impacted", "caries", "periapical", "deep_caries"]
                    cls_id = dis_list.index(
                        filtered_df.iloc[idx]['Disease_Name']
                    )
                    output_img_name = f'{fname_no_ext}_{dis_list[cls_id][0].capitalize()}_{idx}.png'
                else: # It's just for enumeration dataset
                    cls_id = 4
                    output_img_name = f'{fname_no_ext}_{idx}.png'

                if 'train' in name:
                    folder = next(
                        (fol for fol in os.listdir(train_path)
                        if fol.startswith(f"{cls_id}_")),
                        '4_no disease'
                    )
                    fol_class_path = os.path.join(train_path, folder)

                elif 'valid' in name:
                    folder = next(
                        (fol for fol in os.listdir(valid_path)
                        if fol.startswith(f"{cls_id}_")),
                        '4_no disease'
                    )
                    fol_class_path = os.path.join(valid_path, folder)

                else:
                    folder = next(
                        (fol for fol in os.listdir(test_path)
                        if fol.startswith(f"{cls_id}_")),
                        '4_no disease'
                    )
                    fol_class_path = os.path.join(test_path, folder)

                if os.path.exists(os.path.join(fol_class_path,output_img_name)):
                    raise FileExistsError(f'There is files are existed at folder: {fol_class_path} please delete it first to re-preparing again!')

                img = cv2.imread(os.path.join(images_path,fname_no_ext+'.png'))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped_image = crop_image(img,x,y,w,h)

                cv2.imwrite(os.path.join(fol_class_path,output_img_name),cropped_image)


def read_image_and_label(filename_no_ext,data_yaml):
    
    img_path = os.path.join(data_yaml['train'], filename_no_ext + ".png")
    label_path = os.path.join(data_yaml['train'].replace('images','labels'), filename_no_ext + ".txt")

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bboxes = []
    class_labels = []
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            values = line.split()
            cls_id = int(float(values[0]))
            cx, cy, w, h = map(float, values[1:])
            
            bboxes.append([cx, cy, w, h])
            class_labels.append(cls_id)
    
    return image, bboxes, class_labels


def build_transform(img_h, img_w, config, is_disease=False):
    if not is_disease:
        aspect = img_w / img_h
        return A.Compose([
            A.CLAHE(clip_limit=config['clahe_clip_limit'], p=config['clahe_p']),
            A.Rotate(limit=config['rotate_limit'], p=config['rotate_p']),
            A.RandomBrightnessContrast(
                brightness_limit=config['brightness_limit'],
                contrast_limit=config['contrast_limit'],
                p=config['brightness_contrast_p']
            ),
            A.CenterCrop(
                height=int(img_h * config['center_crop_ratio']),
                width=int(img_w * config['center_crop_ratio'])
            ),
            A.RandomResizedCrop(
                size=(img_h, img_w),
                scale=config['resized_crop_scale'],
                ratio=(aspect * config['ratio_margin_low'], aspect * config['ratio_margin_high']),
                p=config['resized_crop_p']
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=config['min_visibility']))
    
    else:
        return A.Compose([
            A.CLAHE(clip_limit=config['clahe_clip_limit'], p=config['clahe_p']),
            A.Rotate(limit=config['rotate_limit'], p=config['rotate_p']),
            A.RandomBrightnessContrast(
                brightness_limit=config['brightness_limit'],
                contrast_limit=config['contrast_limit'],
                p=config['brightness_contrast_p']
            ),
            A.GaussNoise(
                var_limit=config['gauss_noise_var'],
                p=config['gauss_noise_p']
            ),
        ])


def augment_and_save(image, bboxes, class_labels, n_copies, base_filename, output_images, output_labels,
                      aug_config, is_disease=False, debugging=False):
    if not is_disease:
        img_h, img_w = image.shape[:2]
        transform = build_transform(img_h, img_w, aug_config, is_disease)

        for n in range(n_copies):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            new_img = augmented['image']
            new_bboxes = augmented['bboxes']
            new_labels = augmented['class_labels']
            new_filename = f"{base_filename}_aug{n}"

            if debugging:
                visualize_augmentation(new_img, new_bboxes, new_labels, title=new_filename)
            else:
                cv2.imwrite(os.path.join(output_images, f"{new_filename}.png"), new_img)
                with open(os.path.join(output_labels, f"{new_filename}.txt"), 'w') as f:
                    for cls_id, (cx, cy, w, h) in zip(new_labels, new_bboxes):
                        f.write(f"{cls_id}  {cx}  {cy}  {w}  {h}\n")
    
    else:

        transform = build_transform(None,None,aug_config,is_disease=True)
        for n in range(n_copies):
            augmented = transform(image=image)
            new_img = augmented['image']
            new_filename = f"{base_filename}_aug{n}"
            if debugging:
                visualize_augmentation(new_img, None, None, title=new_filename)
            else:
                cv2.imwrite(os.path.join(output_images, f"{new_filename}.png"), new_img)


def apply_smart_aug(data_yaml,aug_config, is_disease=False, apply_debug=False):

    if is_disease:
        
    
    imgs_path = data_yaml['train']
    labels_path = data_yaml['train'].replace('images','labels')
    
    all_files_no_ext = [item.split('.')[0] for item in os.listdir(imgs_path)]

        
    if apply_debug:
        rand_fname = np.random.choice(all_files_no_ext)
        image,bboxes,class_labels = read_image_and_label(rand_fname,data_yaml)
        augment_and_save(image=image,
                        bboxes=bboxes,
                        class_labels=class_labels,
                        n_copies=3,
                        base_filename=rand_fname,
                        output_images=imgs_path,
                        output_labels=labels_path,
                        aug_config=aug_config,
                        debugging=True)

    else:
        aug_files = [f for f in all_files_no_ext if 'aug' in f]
        n_aug = len(aug_files)

        if n_aug > 0:
            print(f"Warning: Found {n_aug} images are already augmented!")
            confirm = input(f'Do you want to delete all {n_aug} augmented images before processing? - (y or n): ').lower().strip()

            if confirm == 'y':
                deleted_count = 0
                failed_count = 0

                for fname in aug_files:
                    img_path = os.path.join(imgs_path, fname + '.png')
                    label_path = os.path.join(labels_path, fname + '.txt')
                    try:
                        os.remove(img_path)
                        os.remove(label_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Failed to remove {fname}: {e}")
                        failed_count += 1

                # clear_output()
                print(f"Deleted {deleted_count} augmented images. Failed: {failed_count}.")
                print("Check the failed deletetion if found and Re-Run the function Please.")
                return

        
        for fname in tqdm(all_files_no_ext,desc=f'Augmenting Images Now...'):

            image,bboxes,class_labels = read_image_and_label(fname,data_yaml)

            if 'Upper Right' in data_yaml['names']:
                n=3
            elif 'Tooth' in data_yaml['names']:
                n_teeth = len(bboxes)

                if n_teeth >= 7:
                    n = 6
                elif n_teeth >= 4:
                    n = 4
                else:
                    n = 2
            elif 'caries' in data_yaml['names']:
                if 2 in class_labels:
                    n = 10
                elif 0 in class_labels or 3 in  class_labels:
                    n = 3
                else:
                    n = 1
            else:
                n=4
                
            augment_and_save(image=image,
                            bboxes=bboxes,
                            class_labels=class_labels,
                            n_copies=n,
                            base_filename=fname,
                            output_images=imgs_path,
                            output_labels=labels_path,
                            aug_config= aug_config)



def export_enum_by_quad_using_model(quadrant_model, enum_images_path, output_root,
                                       enum_df, conf_threshold=0.5,
                                       debugging=False, debug_limit=5):

    debug_count = 0

    for fname in tqdm(enum_df['File_Name'].unique(), desc='Cropping using trained quadrant model'):

        if debugging and debug_count >= debug_limit:
            break

        img_path = os.path.join(enum_images_path, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = quadrant_model.predict(img_path, conf=conf_threshold, verbose=False)[0]

        for box in results.boxes:
            if debugging and debug_count >= debug_limit:
                break

            quad_class_id = int(box.cls[0])
            quad_name = quadrant_model.names[quad_class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            crop_h, crop_w = cropped_img.shape[:2]
            if crop_h == 0 or crop_w == 0:
                continue

            teeth_rows = enum_df[(enum_df['File_Name'] == fname) & (enum_df['Quad'] == quad_name)]

            new_labels = []
            for _, row in teeth_rows.iterrows():
                tx, ty, tw, th = row['Bbox']
                new_x = tx - x1
                new_y = ty - y1
                if new_x + tw <= 0 or new_y + th <= 0 or new_x >= crop_w or new_y >= crop_h:
                    continue
                cx = (new_x + tw / 2) / crop_w
                cy = (new_y + th / 2) / crop_h
                nw = tw / crop_w
                nh = th / crop_h
                cx, cy = np.clip([cx, cy], 0, 1)
                nw, nh = np.clip([nw, nh], 0, 1)
                new_labels.append((row['Enumeration'], cx, cy, nw, nh))

            if not new_labels:
                continue

            base_name = f"{fname.split('.')[0]}_{quad_name.replace(' ', '')}"

            if debugging:
                visualize_augmentation(
                    cropped_img,
                    [(cx, cy, nw, nh) for _, cx, cy, nw, nh in new_labels],
                    [cls_id for cls_id, _, _, _, _ in new_labels],
                    title=base_name
                )
                debug_count += 1
                continue

            cv2.imwrite(os.path.join(output_root, 'images', f"{base_name}.png"),
                        cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            with open(os.path.join(output_root, 'labels', f"{base_name}.txt"), 'w') as f:
                for cls_id, cx, cy, nw, nh in new_labels:
                    f.write(f"{cls_id} {cx} {cy} {nw} {nh}\n")