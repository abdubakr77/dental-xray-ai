import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from sklearn.utils import shuffle

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


def visualize_augmentation(images, bboxes_list=None, class_labels_list=None, titles=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for i, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image, cmap="gray")

        bboxes = bboxes_list[i] if bboxes_list else None
        class_labels = class_labels_list[i] if class_labels_list else None

        if bboxes and class_labels:
            img_h, img_w = image.shape[:2]
            for cls_id, (cx, cy, w, h) in zip(class_labels, bboxes):
                x = (cx - w / 2) * img_w
                y = (cy - h / 2) * img_h
                box_w = w * img_w
                box_h = h * img_h

                rect = patches.Rectangle((x, y), box_w, box_h, linewidth=1.5,
                                          edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 5, str(cls_id), color='red', fontsize=9,
                        bbox=dict(facecolor='black', alpha=0.5, pad=0.5))

        title = titles[i] if titles else f"Image {i}"
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def show_all_images_counts(all_path: list):
    """Count images per class for each split."""
    all_images_counts = {}
    for path in all_path:
        labels = os.listdir(path)
        img_counts = {}
        for label_name in labels:
            label_path = os.path.join(path, label_name)
            img_counts[label_name] = len(os.listdir(label_path))
        all_images_counts[path.split("\\")[-1]] = img_counts
    return all_images_counts

def show_random_image(PATH, num_of_samples=16):
    """Display a 4x4 grid of random samples per class side by side."""
    labels = os.listdir(PATH)
    fig = plt.figure(figsize=(15, 6))
    outer = fig.add_gridspec(1, 2, wspace=0.2)

    for j, label_name in enumerate(labels[:2]):
        label_path = os.path.join(PATH, label_name)
        images = shuffle(os.listdir(label_path))[:num_of_samples]
        inner = outer[j].subgridspec(4, 4)

        ax_title = fig.add_subplot(outer[j])
        ax_title.set_title(label_name, fontsize=14)
        ax_title.axis("off")

        for i, img_name in enumerate(images):
            r, c = divmod(i, 4)
            ax = fig.add_subplot(inner[r, c])
            ax.imshow(plt.imread(os.path.join(label_path, img_name)), cmap="gray")
            ax.axis("off")

    plt.show()

