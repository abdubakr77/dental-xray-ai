import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import os
import shutil
from time import sleep
from tqdm import tqdm

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
    import cv2

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
                    cls_id = ["impacted", "caries", "periapical", "deep_caries"].index(
                        filtered_df.iloc[idx]['Disease_Name']
                    )
                    output_img_name = f'{fname_no_ext}.png'
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

                if len(os.listdir(fol_class_path)) >= 1:
                    raise FileExistsError(f'There is files are existed at folder: {fol_class_path} please delete it first to re-preparing again!')

                img = cv2.imread(os.path.join(images_path,fname_no_ext+'.png'))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped_image = crop_image(img,x,y,w,h)

                cv2.imwrite(os.path.join(fol_class_path,output_img_name),cropped_image)