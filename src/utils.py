import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

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


def show_image_boxes(df,images_path,target:list = None):
    if target and len(target) == 2:
        rows = df[df[target[0]] == target[1]]
        length = len(rows)
        row = rows.iloc[np.random.randint(low=0,high=length)]
        fname = row['File_Name']
        data = rows[rows['File_Name'] == fname]
    elif target is None:
        row = df.iloc[np.random.randint(0,len(df))]
        fname = row['File_Name']
        data = df[df['File_Name'] == fname]
    else:
        raise IndexError(f"Index out of range! Must be only 2. Got {len(target)}")
    

    img = plt.imread(os.path.join(images_path, fname))
    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.imshow(img, cmap="gray")

    for _, row in data.iterrows():
        x, y, w, h = list(row['Bbox'])
        img_h, img_w = img.shape[:2]
        if max(x, y, w, h) <= 1.0:
            x = x * img_w
            y = y * img_h
            w = w * img_w
            h = h * img_h

        color = 'red'

        if 'Disease_Name' in df.columns:

            disease_colors = {
                'caries': 'red',
                'deep_caries': 'orange',
                'periapical': 'blue',
                'impacted': 'purple'
            }

            disease = row['Disease_Name']
            color = disease_colors.get(disease, 'gray')

            legend_elements = [patches.Patch(facecolor='none', edgecolor=c, label=d) 
                                for d, c in disease_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right')

        
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        quad = row['Quad']
        
        label = f'{quad}'

        if 'Enumeration' in df.columns:
            tooth_num = row['Enumeration']

            label = f'{quad} | {tooth_num}'


        ax.text(x, y - 5, label, color=color, fontsize=9,
                bbox=dict(facecolor='black', alpha=0.5, pad=0.5))


    ax.set_title(fname)