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

def show_image(
    df,
    images_path,
    target=None,
    draw_mask=False,
    draw_box=True,
    mask_opacity=0.3
):
    import matplotlib
    # Select image
    if target and len(target) == 2:
        rows = df[df[target[0]] == target[1]]

        if len(rows) == 0:
            raise ValueError(f"No data found for {target[0]} = {target[1]}")

        row = rows.iloc[np.random.randint(len(rows))]
        fname = row['File_Name']
        data = rows[rows['File_Name'] == fname]

    elif target is None:
        row = df.iloc[np.random.randint(len(df))]
        fname = row['File_Name']
        data = df[df['File_Name'] == fname]

    else:
        raise IndexError(
            f"Index out of range! Must be only 2. Got {len(target)}"
        )

    # Read image
    image = cv2.imread(os.path.join(images_path, fname))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 15))


    # Disease colors
    disease_colors = {
        'caries': 'red',
        'deep_caries': 'orange',
        'periapical': 'blue',
        'impacted': 'purple'
    }

    # Draw annotations
    for _, row in data.iterrows():

        color = 'red'

        if 'Disease_Name' in df.columns:
            color = disease_colors.get(
                row['Disease_Name'],
                'gray'
            )

        # Label
        label = str(row['Quad'])

        if 'Enumeration' in df.columns:
            label += f" | {row['Enumeration']}"

        # Mask
        if draw_mask:

            points = np.array(row.iloc[5]).reshape(-1, 2)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)

            rgb = np.array(
                matplotlib.colors.to_rgb(color)
            ) * 255

            image[mask == 255] = (
                image[mask == 255] * (1 - mask_opacity)
                + rgb * mask_opacity
            ).astype(np.uint8)

            xmin = int(points[:, 0].min())
            ymin = int(points[:, 1].min())
            xmax = int(points[:, 0].max())
            ymax = int(points[:, 1].max())

        # Box
        else:

            x, y, w, h = list(row['Bbox'])

            img_h, img_w = image.shape[:2]

            if max(x, y, w, h) <= 1.0:
                x *= img_w
                y *= img_h
                w *= img_w
                h *= img_h

            xmin = int(x)
            ymin = int(y)
            xmax = int(x + w)
            ymax = int(y + h)

        # Box
        if draw_box:

            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )

            ax.add_patch(rect)

        # Label
        ax.text(
            xmin,
            max(ymin - 5, 10),
            label,
            color=color,
            fontsize=9,
            bbox=dict(
                facecolor='black',
                alpha=0.5,
                pad=0.5
            )
        )
        
    ax.imshow(image)
    # Legend
    if 'Disease_Name' in df.columns:

        legend_elements = [
            patches.Patch(
                facecolor='none',
                edgecolor=color,
                label=disease
            )
            for disease, color in disease_colors.items()
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper right'
        )

    ax.set_title(
        f"{fname.split('.')[0]} | Label Counts: {len(data)}"
    )

    ax.axis("off")
    plt.show()

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

def visualize_quadrant_stage(result):
    """Left: full mouth with the 4 quadrant boxes. Right: the 4 quadrant crops stitched together."""
    boxes = [(*box, name) for name, box in result['quadrant_boxes'].items()]
    annotated = draw_infrence_boxes(result['original_image'], boxes)

    quad_names = list(result['quadrant_images'].keys())
    grid = build_image_grid([result['quadrant_images'][q] for q in quad_names], quad_names, ncols=2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(annotated); ax[0].set_title('Quadrant Detection'); ax[0].axis('off')
    ax[1].imshow(grid); ax[1].set_title('Detected Quadrants'); ax[1].axis('off')
    plt.tight_layout()
    return fig

def visualize_teeth_stage(result):
    """One row per quadrant. Col 1 = quadrant with teeth boxes. Col 2 = each tooth crop, labeled by class."""
    quad_names = list(result['teeth_per_quadrant'].keys())
    fig, ax = plt.subplots(len(quad_names), 2, figsize=(14, 5 * len(quad_names)))
    if len(quad_names) == 1:
        ax = ax.reshape(1, 2)

    for i, quad_name in enumerate(quad_names):
        teeth = result['teeth_per_quadrant'][quad_name]
        quad_img = result['quadrant_images'][quad_name]

        boxes = [(*t['box'], t['class_name']) for t in teeth]
        annotated = draw_infrence_boxes(quad_img, boxes, color=(255, 165, 0))
        ax[i, 0].imshow(annotated); ax[i, 0].set_title(quad_name); ax[i, 0].axis('off')

        crop_images = [t['image'] for t in teeth]
        crop_labels = [f"Tooth {t['class_name']}" for t in teeth]
        grid = build_image_grid(crop_images, crop_labels, ncols=4, cell_size=(150, 150))
        ax[i, 1].imshow(grid); ax[i, 1].set_title(f'{quad_name} - Teeth'); ax[i, 1].axis('off')

    plt.tight_layout()
    return fig

def visualize_healthy_unhealthy_stage(result):
    """One figure per quadrant, all its teeth in a grid, labeled tooth number + status."""
    by_quadrant = {}
    for t in result['all_teeth']:
        by_quadrant.setdefault(t['quad_key'], []).append(t)

    figs = []
    for quad_name, teeth in by_quadrant.items():
        images = [t['image'] for t in teeth]
        labels = [f"{t['class_name']} - {t['status']}" for t in teeth]
        grid = build_image_grid(images, labels, ncols=4, cell_size=(150, 150))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(grid); ax.set_title(f'{quad_name} - Healthy / Unhealthy'); ax.axis('off')
        figs.append(fig)
    return figs

def visualize_disease_stage(result):
    """Only the diseased teeth, grouped by quadrant, labeled with the disease type."""
    by_quadrant = {}
    for t in result['diseased_teeth']:
        by_quadrant.setdefault(t['quad_key'], []).append(t)

    figs = []
    for quad_name, teeth in by_quadrant.items():
        images = [t['image'] for t in teeth]
        labels = [f"{t['class_name']} - {t.get('caries_severity') or t.get('disease', 'Unknown')}" for t in teeth]
        grid = build_image_grid(images, labels, ncols=4, cell_size=(150, 150))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(grid); ax.set_title(f'{quad_name} - Disease Type'); ax.axis('off')
        figs.append(fig)
    return figs

def visualize_final_result(result):
    """Full mouth image, boxes only on diseased teeth, labeled with the final disease name."""
    boxes = []
    for t in result['diseased_teeth']:
        qx1, qy1, qx2, qy2 = result['quadrant_boxes'][t['quad_key'].split('_')[-1]]
        tx1, ty1, tx2, ty2 = t['box']  # coordinates relative to the quadrant crop
        # offset the tooth box by the quadrant's own position on the full image
        final_box = (qx1 + tx1, qy1 + ty1, qx1 + tx2, qy1 + ty2)
        final_name = t.get('caries_severity') or t.get('disease', 'Unknown')
        boxes.append((*final_box, f"{t['class_name']} - {final_name}"))

    annotated = draw_infrence_boxes(result['original_image'], boxes, color=(255, 0, 0))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(annotated); ax.set_title('Detected Disease Summary'); ax.axis('off')
    return fig