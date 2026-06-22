from tqdm import tqdm
import os
import cv2
import yaml

# Stage 2
s2_train_path = r'..\Data\Processed\Stage 2 (Disease Classifier)\train'
s2_val_path = r'..\Data\Processed\Stage 2 (Disease Classifier)\val'
s2_test_path = r'..\Data\Processed\Stage 2 (Disease Classifier)\test'

with open(r'../Data/Processed/Stage 1 (Disease Detection)/data.yaml') as f:
    yolo_file = yaml.safe_load(f)


def export_cropped_images(images_path, only_org:bool=False):
    for fname in tqdm(os.listdir(images_path),desc=f'Cropping images now...'):
        
        if only_org:
            if 'aug' not in fname:
                filename_no_ext = fname.split('.')[0]
            else:
                continue
        else:
            filename_no_ext = fname.split('.')[0]

        img_path = os.path.join(images_path, filename_no_ext + ".png")
        label_path = os.path.join(images_path.replace('images','labels'), filename_no_ext + ".txt")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        with open(label_path,'r') as f:
            for line in f.readlines():
                cls_id = str(int(float(line.split()[0])))
                # print(cls_id)
                x1, y1, x2, y2 = map(float,list(line.split()[1:]))
                
                x_pixel = int((x1 - x2 / 2) * img_w)
                y_pixel = int((y1 - y2 / 2) * img_h)
                w_pixel = int(x2 * img_w)
                h_pixel = int(y2 * img_h)

                x_end = x_pixel + w_pixel
                y_end = y_pixel + h_pixel

                cropped_img = img[y_pixel:y_end,x_pixel:x_end]

                fol_class = [fol for fol in os.listdir(s2_train_path) if cls_id in fol][0]

                if 'train' in images_path:
                    cv2.imwrite(os.path.join(s2_train_path,fol_class, f"{filename_no_ext}_cropped.png"), cropped_img)
                elif 'val' in images_path:
                    cv2.imwrite(os.path.join(s2_val_path,fol_class, f"{filename_no_ext}_cropped.png"), cropped_img)
                else:
                    cv2.imwrite(os.path.join(s2_test_path,fol_class, f"{filename_no_ext}_cropped.png"), cropped_img)