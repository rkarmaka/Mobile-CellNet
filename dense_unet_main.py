import sys
sys.path.insert(0, './utils')

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Concatenate, \
                                    Conv2DTranspose, UpSampling2D, Activation, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Nadam

import numpy as np
import pandas as pd
import seaborn as sns
import cv2 as cv

import  os
import modules.models as models
import evals
import func
import load_images
from argparse import Namespace
from datetime import datetime
########################################################################################################################
epochs=150
batch_size=1

args = Namespace()
args.image_shape = (224,224,1)
args.data_format = 'channels_last'

# The network
args.classes = (0, 1)
args.growth_rate = 8
args.blocks = (6, 12, 24, 48, 24, 12, 6)
args.lr = 0.001
args.lr_decay = 0.99


X_train, y_train = load_images.load_images(args.image_shape)

########################################################################################################################
# Load the Dense U-net model for CNN-Edge
model_edge = models.Dense_Unet(args.image_shape,
                               args.blocks,
                               growth_rate=args.growth_rate,
                               lr=args.lr,
                               data_format=args.data_format)

# Load the Dense U-net model for CNN-ROI
#model_roi = models.Dense_Unet(args.image_shape,
#                              args.blocks,
#                              growth_rate=args.growth_rate,
#                              lr=args.lr,
#                              data_format=args.data_format)

nadam = Nadam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#if weights is not None:
#   model_edge.load_weights(weights)

model_edge.compile(optimizer=nadam,
                   loss='binary_crossentropy',
                   metrics=['acc'])
#model_roi.compile(optimizer=nadam,
#                   loss='categorical_crossentropy',
#                   metrics=['categorical_accuracy'])


print(model_edge.summary())
checkpoint_path_edge='results_patches/model_checkpoint_acc_edge/Checkpoint_best'
log_dir = 'logs_patches/dense_unet_roi/' + datetime.now().strftime('%Y%m%d-%H%M%S')
#tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint=ModelCheckpoint(filepath=checkpoint_path_edge,
                           monitor='val_acc',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max')
history_edge=model_edge.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                            verbose=1,validation_split=0.2,
                            callbacks=[checkpoint])


#checkpoint_path_roi='results/model_checkpoint_acc_roi/Checkpoint_best'
#log_dir = 'logs/dense_unet_edge/' + datetime.now().strftime('%Y%m%d-%H%M%S')
#tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#checkpoint=ModelCheckpoint(filepath=checkpoint_path_roi,
#                           monitor='val_acc',
#                           save_best_only=True,
#                           save_weights_only=True,
#                           mode='max')
#history_roi=model_roi.fit(y_train,roi_train,epochs=epochs,batch_size=batch_size,
#                            verbose=1,validation_split=0.2,
#                            callbacks=[checkpoint,tensorboard_callback])


########################################################################################################################
history_unet_edge_df=pd.DataFrame(history_unet_edge.history)
history_unet_edge_df.to_csv('results_patches/model_dense_unet_edge.csv')

#history_unet_roi_df=pd.DataFrame(history_unet_roi.history)
#history_unet_roi_df.to_csv('results/model_roi.csv')
########################################################################################################################
# Performance measure
