from PIL import Image
from skimage.measure import regionprops
import numpy as np
import json
import cv2

data = json.load(open('data/scene-parse-ins/imgCatIds.json'))
categories = data['categories']
remapping = {
    8: 4,  # Car -> Car
    47: 7,  # Boat -> Ship
    49: 4,  # Bus -> Car
    52: 4,  # Truck -> Car
    58: 6,  # Airplane -> Airplane
    64: 4,  # Van -> Car
    65: 7,  # Ship -> Ship
    73: 5,  # Minibike -> Motorbike
    82: 3,  # Bicycle -> Bicycle
}

with open('data/scene-parse-ins/images/validation.txt') as VAL:
    for line in VAL:
        filepath = line.rstrip()
        annpath = filepath.split('.')[0] + '.png'

        
        img = np.array(Image.open(f'data/scene-parse-ins/images/{filepath}'))
        ann = np.array(Image.open(f'data/scene-parse-ins/annotations_instance/{annpath}'))
        h, w, c = img.shape

        cls_label = ann[:, :, 0]
        mask = ann[:, :, 1]
        num_obj = np.bincount(mask.flatten()).shape[0]
        
        if num_obj > 1:
            for idx in range(1, num_obj):
                curr_mask = np.copy(mask)
                curr_label = np.copy(cls_label)

                curr_mask[curr_mask != idx] = 0
                curr_label[curr_mask != idx] = 0
                
                curr_cls = np.argmax(np.bincount(curr_label.flatten())[1:])
                
                if curr_cls in list(remapping.keys()):
                    curr_mask[curr_mask > 0] = 1
                    curr_mask = np.uint8(curr_mask)

                    prop = regionprops(curr_mask)[0]

                    xmin = float(prop.bbox[1]) / w
                    xmax = float(prop.bbox[3]) / w
                    ymin = float(prop.bbox[0]) / h
                    ymax = float(prop.bbox[2]) / h
                    area = prop.area