import sys
from datetime import datetime

sys.path.insert(0, './utils')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import pandas as pd
import models
import func
import load_images

########################################################################################################################
epochs = 500
batch_size = 4

img_size=(224,224,1)
print('Loading data...')
X_train, y_train = load_images.load_images(img_size)
print('Data loading complete...')
########################################################################################################################
model_res_unetpp=models.Res_Unetpp(image_size=img_size)
print(model_res_unetpp.summary())
model_res_unetpp.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',func.dice_coef])
earlystop=EarlyStopping(monitor='val_dice_coef',patience=50,mode='max')
checkpoint_path='results_patches/model_checkpoint_acc_res_unetpp/Checkpoint_best'
log_dir = 'logs_patches/model_checkpoint_acc_res_unetpp' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                           monitor='val_dice_coef',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max')
history_res_unetpp=model_res_unetpp.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                            verbose=1,validation_split=0.2,
                            callbacks=[checkpoint,tensorboard_callback])


########################################################################################################################
history_res_unetpp_df=pd.DataFrame(history_res_unetpp.history)
history_res_unetpp_df.to_csv('results_patches/model_res_unetpp.csv')
########################################################################################################################
# Performance measure