import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

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

def show_curves(model_history,save_dir=os.getcwd()):
    """
    Plot Loss and Accuracy curves for train and validation splits.

    Parameters
    ----------
    model_history : list of dicts — keys: train_loss, valid_loss, train_acc, valid_acc
    """
    train_loss = [x["train_loss"] for x in model_history]
    valid_loss = [x["valid_loss"] for x in model_history]
    train_acc  = [x["train_acc" ] for x in model_history]
    valid_acc  = [x["valid_acc" ] for x in model_history]

    _, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(train_loss, label="Train",  linewidth=2)
    ax[0].plot(valid_loss, label="Validation", linewidth=2, linestyle="--")
    ax[0].set_title("Loss Over Epochs", fontsize=13, fontweight="bold")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(train_acc, label="Train", linewidth=2)
    ax[1].plot(valid_acc, label="Validation", linewidth=2, linestyle="--")
    ax[1].set_title("Accuracy Over Epochs", fontsize=13, fontweight="bold")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'ACC_LOSS_Curves.png'), dpi=300, bbox_inches='tight')
    plt.show()

def show_confusion_matrix(all_labels, all_preds, classes_names:list, save_dir=os.getcwd()):
    """Print classification report and plot confusion matrix."""
    cm = confusion_matrix(np.array(all_labels), np.array(all_preds))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes_names,
                yticklabels=classes_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(classification_report(all_labels, all_preds, target_names=classes_names))

def draw_infrence_boxes(image, boxes_with_labels, color=(0, 255, 0)):
    """boxes_with_labels: list of (x1, y1, x2, y2, label)"""
    img = image.copy()
    for x1, y1, x2, y2, label in boxes_with_labels:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def build_image_grid(images, labels, ncols=2, cell_size=(220, 220)):
    """Stitches multiple images with a label under each into ONE composite image."""
    nrows = int(np.ceil(len(images) / ncols))
    cw, ch = cell_size
    label_h = 28
    grid = np.ones((nrows * (ch + label_h), ncols * cw, 3), dtype=np.uint8) * 255

    for idx, (img, label) in enumerate(zip(images, labels)):
        row, col = idx // ncols, idx % ncols
        resized = cv2.resize(img, cell_size)
        y0, x0 = row * (ch + label_h), col * cw
        grid[y0:y0 + ch, x0:x0 + cw] = resized
        cv2.putText(grid, label, (x0 + 5, y0 + ch + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return grid