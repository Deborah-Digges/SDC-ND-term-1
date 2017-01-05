import cv2
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import random
from scipy import ndimage
import skimage
from utils import pre_process


"""
    Constants used throughout the program
"""
SIMULATOR_HOME = "/home/ubuntu/data/"
DRIVING_LOG_FILE = "driving_log.csv"
DRIVING_LOG_FILE_PATH = os.path.join(SIMULATOR_HOME, DRIVING_LOG_FILE)
IMAGE_PATH = os.path.join(SIMULATOR_HOME, "IMG")

steering_offset = 0.16


driving_log = pd.read_csv(DRIVING_LOG_FILE_PATH)
driving_log.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


def invert(df, drop_zeros=True):
    """
        1. Create a copy of the data frame
        2. Reverse each angle by multiplying by -1
        3. Append "INV" to image path
    """
    inv_df = df.copy()
    
    if(drop_zeros):
        inv_df = inv_df[inv_df.angle != 0]
    
    inv_df.angle *= -1
    inv_df.image += "|INV"
    return inv_df


def get_copy(df, column, angle_offset=0, filter_zeros=False):
    """
        Return a new data frame from the passed df
        where:
        1. column: is one of "left", "right", "center"
        
        The returned df will have column and the angle from the passed df
        
        If an angle offset is passed, it will be added to all rows in the angle column
    """
    res = df.copy()[[column, "steering"]]
    res.columns = ["image", "angle"]
    if(filter_zeros):
        res = res[res.angle != 0]
    res.angle += angle_offset
    return res


# Construct augmented data set

# Center image, angle unchanged
center = get_copy(driving_log, "center")

# Center image, filter zero angles
# invert angle, flip image
center_inverted = invert(center)

# Gaussian blur
center_blur = get_copy(driving_log, "center")
center_blur.image += "|BLUR"

# Gaussian Noise
center_noise = get_copy(driving_log, "center")
center_noise.image += "|NOISE"

center_inv_blur = center_inverted.copy()
center_inv_blur.image += "|BLUR"

center_inv_noise = center_inverted.copy()
center_inv_noise.image += "|NOISE"

# Left image, Add steering offset
left = get_copy(driving_log, "left", steering_offset)
left = left.sample(frac=1).reset_index(drop=True)

left_blur = get_copy(driving_log, "left", steering_offset, filter_zeros=True)
left_blur.image += "|BLUR"

left_noise = get_copy(driving_log, "left", steering_offset, filter_zeros=True)
left_noise.image += "|NOISE"

left_inv = get_copy(driving_log, "left", steering_offset, filter_zeros=True)
left_inv = invert(left_inv)

left_inv_blur = left_inv.copy()
left_inv_blur.image += "|BLUR"

left_inv_noise = left_inv.copy()
left_inv_noise.image += "|NOISE"

# Right image, subtract steering offset
right = get_copy(driving_log, "right", -steering_offset)
right = right.sample(frac=1).reset_index(drop=True)

right_blur = get_copy(driving_log, "right", -steering_offset, filter_zeros=True)
right_blur.image += "|BLUR"

right_noise = get_copy(driving_log, "right", -steering_offset, filter_zeros=True)
right_noise.image += "|NOISE"

right_inv = get_copy(driving_log, "right", -steering_offset, filter_zeros=True)
right_inv = invert(right_inv)

right_inv_blur = right_inv.copy()
right_inv_blur.image += "|BLUR"

right_inv_noise = right_inv.copy()
right_inv_noise.image += "|NOISE"


all_data = pd.concat([center, center_blur, center_noise,
                      center_inverted, center_inv_noise, center_inv_blur,
                      left, left_blur, left_noise,
                      right, right_blur, right_noise,
                      right_inv, right_inv_blur, right_inv_noise
                     ]).reset_index(drop=True)




n = all_data.shape[0]
batch_size = 64
samples_per_epoch = int(n/batch_size)



def get_flipped_image(image):
    """
        returns image which is flipped about the vertical axis
    """
    return cv2.flip(image, 1)



def get_blurred_image(image):
    """
        image_name: file path of the image 
        Performs a gaussian blur on the image and returns it
    """
    return ndimage.gaussian_filter(image, sigma=1)



def get_speckled_image(image):
    return skimage.img_as_ubyte(skimage.util.random_noise(image.astype(np.uint8), mode='gaussian'))



def get_ops(image_name):
    return image_name.split("|")



def get_image(row):
    image_name = row["image"].strip()
    
    ops = get_ops(image_name)
    
    image_name = ops[0]
    
    ops = ops[1:]
    
    image = cv2.imread(os.path.join(SIMULATOR_HOME, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for op in ops:
        if op == "INV":
            image = get_flipped_image(image)

        elif op == "BLUR":
            image = get_blurred_image(image)

        elif op == "NOISE":
            image = get_speckled_image(image)
            
    return image



def data_generator(df, batch_size=128):
    """
        yields a pair (X, Y) where X and Y are both numpy arrays of length `batch_size`
    """
    n_rows = df.shape[0]
    while True:
        # Shuffle the data frame rows after every complete cycle through the data
        df = df.sample(frac=1).reset_index(drop=True)
        
        for index in range(0, n_rows, batch_size):
            df_batch = df[index: index + batch_size]

            # Ignoring the last batch which is smaller than the requested batch size
            if(df_batch.shape[0] == batch_size):
                X_batch = np.array([pre_process(get_image(row)) for i, row in df_batch.iterrows()])
                y_batch = np.array([row['angle'] for i, row in df_batch.iterrows()])
                yield X_batch, y_batch



# Define the model
input_shape = (X_batch.shape[1], X_batch.shape[2], X_batch.shape[3])
pool_size = (2,2)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=input_shape,
            output_shape=input_shape))
model.add(Convolution2D(32, 3, 3, border_mode='valid', name='conv1', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(32, 3 , 3, border_mode='valid', name='conv2', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3 , 3, border_mode='valid', name='conv3', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', name='conv4', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Convolution2D(128, 3, 3, border_mode='valid', name='conv5', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(128, 3, 3, border_mode='valid', name='conv6', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512,name='hidden1', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(64,name='hidden2', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(16,name='hidden3',init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(1, name='output', init='he_normal'))


adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse")


# Train the model
gen = data_generator(all_data, batch_size=batch_size)
model.fit_generator(gen, samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=2)

# Save the model
model.save_weights("model.h5")
file = open("model.json", "w")
file.write(model.to_json())
file.close()