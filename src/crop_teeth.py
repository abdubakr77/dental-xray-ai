from tqdm import tqdm
import os
import cv2

def export_cropped_images(yolo_images_path:str, output_root:str, only_org:bool=False):
    for fname in tqdm(os.listdir(yolo_images_path),desc=f'Cropping images now...'):
        if only_org:
            if 'aug' not in fname:
                filename_no_ext = fname.split('.')[0]
            else:
                continue
        else:
            filename_no_ext = fname.split('.')[0]
        
        for fol in os.listdir(os.path.join(output_root,'train')):
            if os.path.exists(os.path.join(output_root,'train',fol, f"{filename_no_ext}_cropped.png")):
                continue
            
            img_path = os.path.join(yolo_images_path, filename_no_ext + ".png")
            label_path = os.path.join(yolo_images_path.replace('images','labels'), filename_no_ext + ".txt")

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

                    try:
                        fol_class = [fol for fol in os.listdir(os.path.join(output_root,'train')) if cls_id in fol][0]
                    except:
                        raise FileNotFoundError(f'Classes Folders Are Not Found! Should be the classes folders in each train, valid, and test folders in the output_root path contains: (class_id)_(disease_name)\nEx: 0_Caries,1_Deep_Caries,....')

                    if 'train' in yolo_images_path:
                        cv2.imwrite(os.path.join(output_root,'train',fol_class, f"{filename_no_ext}_cropped.png"), cropped_img)
                    elif 'valid' in yolo_images_path:
                        cv2.imwrite(os.path.join(output_root,'valid',fol_class, f"{filename_no_ext}_cropped.png"), cropped_img)
                    else:
                        cv2.imwrite(os.path.join(output_root,'test',fol_class, f"{filename_no_ext}_cropped.png"), cropped_img)