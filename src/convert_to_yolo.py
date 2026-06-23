import os
import shutil
from time import sleep
from tqdm import tqdm


def export_yolo_dataset(diagnosis_map:dict, images_path:str ,output_root: str, train_df, valid_df=None, test_df=None):
    print("WARNING: Please Check all txt files in labels folder are cleared because this function is recommended to run it only once.\nYou Have 5 seconds from now if you need to stop the code!")
    sleep(5)
    # clear_output(wait=True)

    ds_partitions = {'train_df':train_df,
                     'valid_df':valid_df,
                     'test_df':test_df,}
    

    for name,df in ds_partitions.items():
        for idx in tqdm(range(len(df)),f'{name} Is Processing Now...'):
            file_name = df.iloc[idx]['File_Name']
            x, y, w, h = df.iloc[idx]['Bbox']
            img_h = df.iloc[idx]['Height']
            img_w = df.iloc[idx]['Width']

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

            cls_id = list(diagnosis_map.values()).index(df.iloc[idx]['Disease_Name'])

            if name in 'train' in name: path = os.path.join(output_root,'train','labels',file_name.replace('png','txt'))
            elif'valid' in name: path = os.path.join(output_root,'valid','labels',file_name.replace('png','txt'))
            else : path = os.path.join(output_root,'test','labels',file_name.replace('png','txt'))

            if os.path.exists(path):
                raise FileExistsError(f"Error: There is a files are existed...\nIf you need to ignore it then type (None) in {name} parameter to not duplicate the files!\nOr make sure that you deleted the files")


            with open(path,'a') as f:
                f.write(f'{cls_id}  {x1}  {y1}  {x2}  {y2}\n')

            shutil.copy2(os.path.join(images_path,file_name),path.replace('labels','images').replace('txt','png'))
            # break 
        print("Done!")