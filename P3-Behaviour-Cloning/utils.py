import cv2
import numpy as np

def pre_process(image):
    """
        image: input image
        
        - convert to grayscale
        - resize image to half it's original size
    """
    
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    #image = np.mean(image, axis=2)
    
#     data_min = np.min(image)
#     data_max = np.max(image)
#     b = 0.5
#     a = -0.5
#     image = ((b-a) * (image - data_min)/(data_max - data_min)) + a
    
    rows_to_crop_top = int(image.shape[0] * 0.35)
    rows_to_crop_bottom = int(image.shape[0] * 0.1)
    image = image[rows_to_crop_top:image.shape[0] - rows_to_crop_bottom, :]    
    
#     row,col= image.shape
#     gauss = np.random.normal(mean,sigma,(row,col))
#     image = image + gauss
    #image =  cv2.normalize(cropped_image, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return cv2.resize(image, (0,0), fx=0.5, fy=0.5) 

def flatten(image, num_channels):
    return image.reshape((image.shape[0], image.shape[1], num_channels))

def get_pre_processed_image(image, num_channels):
    return flatten(pre_process(image), num_channels)