from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Add, Activation, BatchNormalization, DepthwiseConv2D

from skimage.io import imread_collection
import numpy as np
import cv2 as cv
import os

def load_test_data(path, im_dim, out_dim, flag=0):
    x=imread_collection(os.path.join(path,'*.jpg'))
    if flag==1:
        data=np.zeros([len(x.files),out_dim[0],out_dim[1],1],dtype='uint8')
    else:
        data=np.zeros([len(x.files),out_dim[0],out_dim[1]],dtype='uint8')
    for i,fname in enumerate(x.files):
        data[i,:im_dim[0],:im_dim[1]]=cv.imread(fname,flags=flag)
    
    return data



def mobile_bottleneck(x, expand_filters, contract_filters, strides=1, add=True):
    c = Conv2D(expand_filters, (1, 1), strides=1, activation=None, padding='same')(x)
    c = BatchNormalization()(c)
    c = Activation(tf.nn.relu6)(c)
    # c=Activation('relu')(c)
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', activation=None)(c)
    c = BatchNormalization()(c)
    c = Activation(tf.nn.relu6)(c)
    # c=Activation('relu')(c)
    c = Conv2D(contract_filters, (1, 1), strides=1, padding='same', activation=None)(c)
    c = BatchNormalization()(c)
    if (strides == 1) and (add):
        c = Add()([x, c])
    return c

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def read_image_file_names(root_folder):
    image_files = []

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                image_files.append(file_path)

    return image_files



def pad_image(image, out_dim):
    h,w = image.shape
    out_image = np.zeros((out_dim[0], out_dim[1]))
    out_image[:h,:w] = image

    return out_image
