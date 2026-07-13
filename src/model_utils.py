import cv2
import matplotlib.pyplot as plt
import numpy as np

def predict(yolo_model, image, conf_filter=0.5, custom_draw_box=None, crop_output_image:str=False, save_output:bool=False, save_dir:str=None):
    outputs = yolo_model.predict(image,conf=conf_filter)

    output = outputs[0]
    boxes = output.boxes
    names = output.names

    _,ax = plt.subplots(1,2,figsize=(12,8))
    for i in range(len(boxes)):
        confidence = boxes[i].conf
        coordinates = boxes[i].xyxy
        cls_name = names[boxes[i].cls.numpy().item()]

        print(f'Class Name: {cls_name}')
        print(f'Coordinates: {coordinates}')
        print(f'Confidence: %{(confidence * 100):.4}')

    ax[0].set_title('Original Image')
    ax[0].imshow(image)
    ax[0].yticks([])
    ax[0].xticks([])

    ax[1].set_title('Object Detected')
    ax[1].imshow(output.plot()[:,:,::-1])
    ax[1].yticks([])
    ax[1].xticks([])
    
    plt.tight_layout()
    plt.show()