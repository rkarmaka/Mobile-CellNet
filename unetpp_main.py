import sys
sys.path.insert(0, './utils')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam
import pandas as pd
import models
import func
import load_images
from datetime import datetime


########################################################################################################################
epochs = 500
batch_size = 4

img_size=(224,224,1)
print('Loading data...')
X_train, y_train = load_images.load_images(img_size,roi=True)
print('Data loading complete...')
########################################################################################################################
model_unetpp=models.Unetpp(image_size=img_size)
print(model_unetpp.summary())
model_unetpp.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',func.dice_coef])
earlystop=EarlyStopping(monitor='val_dice_coef',patience=50,mode='max')
checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_unetpp_roi/Checkpoint_best'
log_dir = 'new_test_set/logs_patches/model_checkpoint_acc_unetpp_roi' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = CSVLogger('training.log')
checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                           monitor='val_dice_coef',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max')
history_unetpp=model_unetpp.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                            verbose=1,validation_split=0.2,
                            callbacks=[checkpoint,tensorboard_callback])


########################################################################################################################
history_unetpp_df=pd.DataFrame(history_unetpp.history)
history_unetpp_df.to_csv('new_test_set/results_patches/model_unetpp_roi.csv')

########################################################################################################################
# Performance measure
