import sys
sys.path.insert(0, './utils')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import pandas as pd
import modules.models as models
import func
import load_images
from datetime import datetime
########################################################################################################################
epochs = 500
batch_size = 16

img_size=(224,224,1)
print('Loading data...')
X_train, y_train = load_images.load_images(img_size)
print('Data loading complete...')
########################################################################################################################
model_segnet=models.SegNet(image_size=img_size)
print(model_segnet.summary())
model_segnet.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',func.dice_coef])
earlystop=EarlyStopping(monitor='val_dice_coef',patience=50,mode='max')
log_dir = 'logs_patches/model_checkpoint_acc_segnet/' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_path='results_patches/model_checkpoint_acc_segnet/Checkpoint_best'
checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                           monitor='val_dice_coef',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max')
history_segnet=model_segnet.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                            verbose=1,validation_split=0.2,
                            callbacks=[earlystop,checkpoint,tensorboard_callback])


########################################################################################################################
history_segnet_df=pd.DataFrame(history_segnet.history)
history_segnet_df.to_csv('results_patches/model_segnet.csv')

########################################################################################################################
# Performance measure
