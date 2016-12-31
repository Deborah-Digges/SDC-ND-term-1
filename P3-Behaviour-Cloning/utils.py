import cv2
import numpy as np

def pre_process(image, top_prop=0.35, bottom_prop=0.1):
    """
        - Crop the top `top_prop` and the bottom `bottom_prop` of the image
        - Resize the image to half of it's original size
    """
    rows_to_crop_top = int(image.shape[0] * 0.5)
    rows_to_crop_bottom = int(image.shape[0] * 0.1)
    image = image[rows_to_crop_top:image.shape[0] - rows_to_crop_bottom, :]    

    return cv2.resize(image, (0,0), fx=0.5, fy=0.5)