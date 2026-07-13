import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_corner_box(img, x1, y1, x2, y2, label_name, confidence, color=(0, 255, 0), length=35, thickness=3):
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
    text = f"{label_name} {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)



def smart_predict(yolo_model, images_path, conf_filter=0.3, custom_draw_box=None, crop_output_image:str=False, save_output:bool=False, save_dir:str=None):

    if not os.path.exists(images_path):
        raise FileNotFoundError('Image Path not existed! Please Check the path is correct')
    
    rand_image_path = os.path.join(images_path,np.random.choice(os.listdir(images_path)))

    outputs = yolo_model.predict(rand_image_path,conf=conf_filter)

    output = outputs[0]
    boxes = output.boxes
    names = output.names

    _,ax = plt.subplots(1,2,figsize=(18,12))
    for i in range(len(boxes)):
        confidence = np.round(boxes.conf[i].item(),2)
        coordinates = boxes.xyxy[i].tolist()
        cls_name = names[boxes.cls[i].item()]

        print(f'Class Name: {cls_name}')
        print(f'Coordinates: {coordinates}')
        print(f'Confidence: %{(confidence * 100):.4}')

    ax[0].set_title('Original Image')
    ax[0].imshow(cv2.cvtColor(cv2.imread(rand_image_path),cv2.COLOR_BGR2RGB))
    ax[0].axis('off')

    ax[1].set_title('Object Detected')
    ax[1].imshow(output.plot()[:,:,::-1])
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()