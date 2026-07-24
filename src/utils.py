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
import numpy as np 

# pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

diagnosis_map = {
    0: "impacted",
    1: "caries",
    2: "periapical",
    3: "deep_caries"
}

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

def find_duplicate_boxes(df, target_col, iou_threshold=0.9):
    duplicates = []
    
    for fname in df['File_Name'].unique():
        rows = df[df['File_Name'] == fname].reset_index()
        n = len(rows)
        
        for i in range(n):
            for j in range(i+1, n):
                box1 = rows.iloc[i]['Bbox']
                box2 = rows.iloc[j]['Bbox']
                iou = compute_iou(box1, box2)
                text = f'{rows.iloc[i][target_col]} {rows.iloc[j][target_col]}'
                sorted_text = sorted(text.split())
                if iou >= iou_threshold:
                    duplicates.append({
                        'fname': fname,
                        'index_1': rows.iloc[i]['index'],
                        'index_2': rows.iloc[j]['index'],
                        'iou': iou,
                        target_col: f'{sorted_text[0]} & {sorted_text[1]}',
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



def convert_to_yolo(target_col: str, images_path: str, output_root: str,
                     train_df, valid_df, test_df=None, clear_existing=True):

    ds_partitions = {'train_df': train_df,
                      'valid_df': valid_df,
                      'test_df': test_df}

    # ---- check + clear existing images/labels ----
    if clear_existing:
        any_existing = False
        for name, df in ds_partitions.items():
            if df is None:
                continue
            split_folder = name.replace('_df', '')
            imgs_path = os.path.join(output_root, split_folder, 'images')
            labels_path = os.path.join(output_root, split_folder, 'labels')
            if (os.path.exists(imgs_path) and len(os.listdir(imgs_path)) > 0) or \
               (os.path.exists(labels_path) and len(os.listdir(labels_path)) > 0):
                any_existing = True
                break

        if any_existing:
            print("Warning: Found existing images/labels in the YOLO dataset.")
            confirm = input("Do you want to delete them all before re-converting? - (y or n): ").lower().strip()

            if confirm == 'y':
                deleted_count = 0
                failed_count = 0
                for name, df in ds_partitions.items():
                    if df is None:
                        continue
                    split_folder = name.replace('_df', '')
                    for sub in ['images', 'labels']:
                        sub_path = os.path.join(output_root, split_folder, sub)
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
                print("Skipped clearing. Existing files will cause FileExistsError if duplicated.")

    # ---- Real Converting Here ----
    for name, df in ds_partitions.items():
        if df is None:
            continue

        for fname in tqdm(df['File_Name'].unique().tolist(), f'{name} Is Processing Now...'):

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

                if target_col == 'Disease_Name':
                    cls_id = ["impacted", "caries", "periapical", "deep_caries"].index(filtered_df.iloc[idx][target_col])
                elif target_col == 'Enumeration':
                    cls_id = 0
                    if 'Disease_Name' in df.columns:
                        output_img_name_no_ext = f"{fname.split('.')[0]}_dis_teeth"
                else:
                    cls_id = ["Upper Right", "Upper Left", "Lower Left", "Lower Right"].index(filtered_df.iloc[idx][target_col])

                if 'train' in name:
                    labels_path = os.path.join(output_root, 'train', 'labels', output_img_name_no_ext + '.txt')
                    imgs_path = os.path.join(output_root, 'train', 'images', output_img_name_no_ext + '.png')
                elif 'valid' in name:
                    labels_path = os.path.join(output_root, 'valid', 'labels', output_img_name_no_ext + '.txt')
                    imgs_path = os.path.join(output_root, 'valid', 'images', output_img_name_no_ext + '.png')
                else:
                    labels_path = os.path.join(output_root, 'test', 'labels', output_img_name_no_ext + '.txt')
                    imgs_path = os.path.join(output_root, 'test', 'images', output_img_name_no_ext + '.png')

                with open(labels_path, 'a') as f:
                    f.write(f'{cls_id}  {x1}  {y1}  {x2}  {y2}\n')

            if os.path.exists(imgs_path):
                raise FileExistsError(
                    f"Error: File already exists at {imgs_path}.\n"
                    f"Set clear_existing=True and re-run, or delete manually first."
                )

            shutil.copy2(os.path.join(images_path, fname), imgs_path)

        print(f"{name} Done!")


def crop_image(image,x,y,w,h):
    img_h, img_w = image.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    return image[y1:y2, x1:x2]



def prepare_disease_classifier(images_path, output_root, train_df, valid_df, test_df=None,
                                 clear_existing=True):

    ds_partitions = {'train_df': train_df,
                      'valid_df': valid_df,
                      'test_df': test_df}

    dis_list = ["impacted", "caries", "periapical", "deep_caries"]

    # ---- check + clear existing images ----
    if clear_existing:
        any_existing = False
        for name, df in ds_partitions.items():
            if df is None:
                continue
            split_folder = name.replace('_df', '')  # train_df -> train
            split_path = os.path.join(output_root, split_folder)
            if os.path.exists(split_path):
                for fol in os.listdir(split_path):
                    fol_path = os.path.join(split_path, fol)
                    if os.path.isdir(fol_path) and len(os.listdir(fol_path)) > 0:
                        any_existing = True
                        break

        if any_existing:
            print("Warning: Found existing images in the disease classifier dataset.")
            confirm = input("Do you want to delete them all before re-preparing? - (y or n): ").lower().strip()

            if confirm == 'y':
                deleted_count = 0
                failed_count = 0
                for name, df in ds_partitions.items():
                    if df is None:
                        continue
                    split_folder = name.replace('_df', '')
                    split_path = os.path.join(output_root, split_folder)
                    if not os.path.exists(split_path):
                        continue
                    for fol in os.listdir(split_path):
                        fol_path = os.path.join(split_path, fol)
                        if not os.path.isdir(fol_path):
                            continue
                        for f in os.listdir(fol_path):
                            try:
                                os.remove(os.path.join(fol_path, f))
                                deleted_count += 1
                            except Exception as e:
                                print(f"Failed to remove {f}: {e}")
                                failed_count += 1

                print(f"Deleted {deleted_count} images. Failed: {failed_count}.")
            else:
                print("Skipped clearing. Existing images will cause FileExistsError if duplicated.")

    for name, df in ds_partitions.items():
        if df is None:
            continue
        for fname in tqdm(df['File_Name'].unique().tolist(), f'{name} Is Processing Now...'):

            filtered_df = df[df['File_Name'] == fname]
            fname_no_ext = f"{fname.split('.')[0]}"

            for idx in range(len(filtered_df)):

                x, y, w, h = filtered_df.iloc[idx]['Bbox']

                train_path = os.path.join(output_root, 'train')
                valid_path = os.path.join(output_root, 'valid')
                test_path = os.path.join(output_root, 'test')

                if 'Disease_Name' in df.columns:
                    cls_id = dis_list.index(filtered_df.iloc[idx]['Disease_Name'])
                    output_img_name = f'{fname_no_ext}_{dis_list[cls_id][0].capitalize()}_{idx}.png'
                else:
                    cls_id = 4
                    output_img_name = f'{fname_no_ext}_{idx}.png'

                if 'train' in name:
                    folder = next((fol for fol in os.listdir(train_path) if fol.startswith(f"{cls_id}_")), '4_no disease')
                    fol_class_path = os.path.join(train_path, folder)
                elif 'valid' in name:
                    folder = next((fol for fol in os.listdir(valid_path) if fol.startswith(f"{cls_id}_")), '4_no disease')
                    fol_class_path = os.path.join(valid_path, folder)
                else:
                    folder = next((fol for fol in os.listdir(test_path) if fol.startswith(f"{cls_id}_")), '4_no disease')
                    fol_class_path = os.path.join(test_path, folder)

                output_path = os.path.join(fol_class_path, output_img_name)
                if os.path.exists(output_path):
                    raise FileExistsError(
                        f'File already exists at: {output_path}. '
                        f'Set clear_existing=True and re-run, or delete manually first.')

                img = cv2.imread(os.path.join(images_path, fname_no_ext + '.png'))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped_image = crop_image(img, x, y, w, h)

                cv2.imwrite(output_path, cropped_image)


def read_image_and_label(filename_no_ext,data_yaml):
    
    img_path = os.path.join(data_yaml['train'], filename_no_ext + ".png")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if 'Caries' in data_yaml['names']:
        return image
    
    label_path = os.path.join(data_yaml['train'].replace('images','labels'), filename_no_ext + ".txt")
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
            A.HorizontalFlip(p=config['hflip_p']),
            A.VerticalFlip(p=config['vflip_p']),
            A.Rotate(limit=config['rotate_limit'], p=config['rotate_p']),
            A.RandomBrightnessContrast(
                brightness_limit=config.get('brightness_limit', 0.1),
                contrast_limit=config.get('contrast_limit', 0.1),
                p=config.get('brightness_contrast_p', 0.35)
            ),
            A.CLAHE(clip_limit=config.get('clahe_clip_limit', 2.0), p=config.get('clahe_p', 0.5)),
            A.RandomScale(scale_limit=config['zoom_scale'], p=config['zoom_p']),
            A.Blur(blur_limit=config.get('blur_var', (3, 5)), p=config.get('blur_p', 0.15)),
        ])


def augment_and_save(image, bboxes, class_labels, n_copies, base_filename, output_images, output_labels,
                      aug_config, is_disease=False, debugging=False):
    
    all_images,all_bboxes,all_labels,all_filenames = [],[],[],[]

    if not is_disease:
        img_h, img_w = image.shape[:2]
        transform = build_transform(img_h, img_w, aug_config, is_disease)
        
        for n in range(n_copies):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            new_img = augmented['image']
            new_bboxes = augmented['bboxes']
            new_labels = augmented['class_labels']
            new_filename = f"{base_filename}_aug{n}"

            all_images.append(new_img)
            all_bboxes.append(new_bboxes)
            all_labels.append(new_labels)
            all_filenames.append(new_filename)
        
        if debugging:
            visualize_augmentation(all_images, all_bboxes, all_labels, titles=all_filenames)
        else:
            for new_img,new_bboxes,new_labels,new_filename in zip(all_images,all_bboxes,all_labels,all_filenames):
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

            all_images.append(new_img) ; all_filenames.append(new_filename)

        if debugging:
            visualize_augmentation(all_images, None, None, titles=all_filenames)
        else:
            for new_img,new_filename in zip(all_images,all_filenames):
                cv2.imwrite(os.path.join(output_images, f"{new_filename}.png"), new_img)


def apply_smart_aug(data_yaml,aug_config, is_disease=False, apply_debug=False):

    main_images_path = data_yaml['train']

    if is_disease:
        disease_images_per_class = {}
        all_fol_classes = os.listdir(main_images_path)
        aug_exists = False
        for fol_class in all_fol_classes:
            fol_class_path = os.path.join(main_images_path, fol_class)
            all_files_no_ext = [item.split('.')[0] for item in os.listdir(fol_class_path)]
            disease_images_per_class[fol_class] = all_files_no_ext
            if any('aug' in f for f in all_files_no_ext):
                aug_exists = True
    else:
        labels_path = main_images_path.replace('images', 'labels')
        all_files_no_ext = [item.split('.')[0] for item in os.listdir(main_images_path)]
        aug_exists = any('aug' in f for f in all_files_no_ext)
        

    if aug_exists and not apply_debug:
        print("Warning: Found existing augmented images in this dataset.")
        confirm = input("Do you want to clear them now using clear_dataset_images()? - (y or n): ").lower().strip()
        if confirm == 'y':
            clear_dataset_images(data_yaml, is_disease=is_disease, target='augmented', confirm_prompt=False)
        else:
            print("Skipped clearing. Proceeding with existing augmented images still in place.")



    # Debugging HERE
    if apply_debug:
        
        if is_disease:
            rand_class = np.random.choice(list(disease_images_per_class.keys()))
            rand_fname = np.random.choice(disease_images_per_class[rand_class])
            image = read_image_and_label(os.path.join(rand_class,rand_fname),data_yaml)
            augment_and_save(image=image,
                            bboxes=None,
                            class_labels=None,
                            n_copies=3,
                            base_filename=rand_fname,
                            output_images=None,
                            output_labels=None,
                            aug_config=aug_config,
                            debugging=True,is_disease=True)
        else:
            rand_fname = np.random.choice(all_files_no_ext)
            image,bboxes,class_labels = read_image_and_label(rand_fname,data_yaml)
            augment_and_save(image=image,
                            bboxes=bboxes,
                            class_labels=class_labels,
                            n_copies=3,
                            base_filename=rand_fname,
                            output_images=None,
                            output_labels=None,
                            aug_config=aug_config,
                            debugging=True)
            
        return


    if is_disease:
        for fol_class in list(disease_images_per_class.keys()):
            class_label = int(fol_class[0])
            if class_label == 4:
                continue

            for fname in tqdm(disease_images_per_class[fol_class],
                               desc=f'Augmenting {diagnosis_map[int(fol_class[0])]} Images Now...'):
                image = read_image_and_label(os.path.join(fol_class, fname), data_yaml)

                if class_label == 2:
                    n = 15
                elif class_label == 0 or class_label == 3:
                    n = 4
                else:
                    if np.random.choice([0, 1]):
                        continue
                    n = 1

                augment_and_save(image=image, bboxes=None, class_labels=None, n_copies=n,
                                  base_filename=fname, output_images=os.path.join(main_images_path, fol_class),
                                  output_labels=None, aug_config=aug_config, is_disease=True)
                    
    else:
        for fname in tqdm(all_files_no_ext, desc='Augmenting Images Now...'):
            image, bboxes, class_labels = read_image_and_label(fname, data_yaml)

            if 'Upper Right' in data_yaml['names']:
                n = 3
            elif 'Tooth' in data_yaml['names'] or data_yaml['nc'] == 8:
                n_teeth = len(bboxes)
                n = 6 if n_teeth <= 4 else 4 if n_teeth <= 6 else 3
            elif 'caries' in data_yaml['names']:
                if 2 in class_labels:
                    n = 10
                elif 0 in class_labels or 3 in class_labels:
                    n = 3
                else:
                    n = 1

            augment_and_save(image=image, bboxes=bboxes, class_labels=class_labels, n_copies=n,
                              base_filename=fname, output_images=main_images_path,
                              output_labels=labels_path, aug_config=aug_config)


def clear_dataset_images(data_yaml, is_disease=False, target='augmented', confirm_prompt=True):
    """
    Deletes images (and matching label files, if applicable) from the dataset.

    Parameters:
        data_yaml: dict with 'train' key pointing to the images path
        is_disease: True for classifier-style folders (class subfolders, no label files)
                    False for YOLO-style (flat images/ + labels/)
        target: 'augmented' -> only files with 'aug' in the name
                'original'  -> only files WITHOUT 'aug' in the name
                'all'       -> everything
        confirm_prompt: if True, asks for y/n confirmation before deleting
    """

    main_images_path = data_yaml['train']

    def matches_target(fname):
        if target == 'augmented':
            return 'aug' in fname
        elif target == 'original':
            return 'aug' not in fname
        elif target == 'all':
            return True
        else:
            raise ValueError(f"Invalid target: {target}. Use 'augmented', 'original', or 'all'.")

    # ---- collect files to delete ----
    files_to_delete = []

    if is_disease:
        all_fol_classes = os.listdir(main_images_path)
        for fol_class in all_fol_classes:
            fol_class_path = os.path.join(main_images_path, fol_class)
            all_files_no_ext = [item.split('.')[0] for item in os.listdir(fol_class_path)]
            files_to_delete += [
                os.path.join(fol_class_path, f + '.png')
                for f in all_files_no_ext if matches_target(f)
            ]
    else:
        all_files_no_ext = [item.split('.')[0] for item in os.listdir(main_images_path)]
        files_to_delete = [
            os.path.join(main_images_path, f + '.png')
            for f in all_files_no_ext if matches_target(f)
        ]

    n_files = len(files_to_delete)

    if n_files == 0:
        print(f"No '{target}' images found. Nothing to delete.")
        return

    print(f"Found {n_files} images matching target='{target}'.")

    if confirm_prompt:
        confirm = input(f"Delete all {n_files} images? - (y or n): ").lower().strip()
        if confirm != 'y':
            print("Cancelled. No files deleted.")
            return

    # ---- delete ----
    deleted_count = 0
    failed_count = 0

    for fpath in tqdm(files_to_delete, desc=f'Deleting {target} images...'):
        try:
            os.remove(fpath)

            if not is_disease:
                label_path = fpath.replace('images', 'labels').replace('.png', '.txt')
                if os.path.exists(label_path):
                    os.remove(label_path)

            deleted_count += 1
        except Exception as e:
            print(f"Failed to remove {fpath}: {e}")
            failed_count += 1

    print(f"Deleted {deleted_count} images. Failed: {failed_count}.")
    if failed_count > 0:
        print("Check the failed deletions above and re-run if needed.")