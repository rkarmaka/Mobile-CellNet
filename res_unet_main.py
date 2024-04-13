import sys
sys.path.insert(0, './utils')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import seaborn as sns
import cv2 as cv

import  os
import models
import evals
import func
import load_images
from datetime import  datetime

########################################################################################################################
print("Running residual U-Net no RoI for modified dataset!")
np.random.seed(1)
epochs = 500
batch_size = 8
img_size=(224,224,1)

X_train, y_train = load_images.load_images(img_size, roi=True)

print('Data loading complete...')
########################################################################################################################
model_res_unet=models.Res_Unet(image_size=img_size)
print(model_res_unet.summary())
model_res_unet.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',func.dice_coef])
earlystop=EarlyStopping(monitor='val_dice_coef',patience=50,mode='max')
checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_res_unet_roi/Checkpoint_best'
log_dir = 'new_test_set/logs_patches/model_checkpoint_acc_res_unet_roi' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                           monitor='val_dice_coef',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max')
history_res_unet=model_res_unet.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                            verbose=1,validation_split=0.2,
                            callbacks=[checkpoint,tensorboard_callback])


########################################################################################################################
history_res_unet_df=pd.DataFrame(history_res_unet.history)
history_res_unet_df.to_csv('new_test_set/results_patches/model_res_unet_roi.csv')

########################################################################################################################
# Performance measure
