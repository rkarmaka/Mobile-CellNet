import sys
sys.path.insert(0, './utils')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import pandas as pd
import models
import func
import load_images
from datetime import datetime

########################################################################################################################
epochs = 500
batch_size = 8

img_size=(224,224,1)
#print('Loading data...')

#X_train, y_train = load_images.load_images(img_size,roi=False)
#print(X_train.shape)
#print('Data loading complete...')

########################################################################################################################
print('Loading data...')

x_path='D:\\Michigan Tech\\Fun\\Eye\\Experiments\\2. U-Net\\New_dataset\\train\\train\\main'
y_path='D:\\Michigan Tech\\Fun\\Eye\\Experiments\\2. U-Net\\New_dataset\\train\\train\\m'
y_path_roi='D:\\Michigan Tech\\Fun\\Eye\\Experiments\\2. U-Net\\New_dataset\\train\\train\\roi'

X_train = load_images.load_x_data(x_path)
y_train = load_images.load_y_data([y_path,y_path_roi])
print(X_train.shape)
print('Data loading complete...')

########################################################################################################################
model_unet=models.Unet_new(image_size=img_size)
print(model_unet.summary())
model_unet.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',func.dice_coef])
earlystop=EarlyStopping(monitor='val_dice_coef',patience=50,mode='max')
checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_unet_2out/Checkpoint_best'
log_dir = 'new_test_set/logs_patches/model_checkpoint_acc_unet_2out' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                           monitor='val_dice_coef',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max')
history_unet=model_unet.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                            verbose=1,validation_split=0.2,
                            callbacks=[checkpoint,tensorboard_callback,earlystop])


########################################################################################################################
history_unet_df=pd.DataFrame(history_unet.history)
history_unet_df.to_csv('new_test_set/results_patches/model_acc_unet_2out.csv')

########################################################################################################################
# Performance measure
