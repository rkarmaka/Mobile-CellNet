import sys
sys.path.insert(0, './utils')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, MeanIoU, MeanAbsoluteError
import pandas as pd
import modules.models as models
import func
import numpy as np
import load_images
from datetime import datetime

########################################################################################################################
img_size=448

# UNet
model=models.Unet(image_size=(img_size,img_size,1))
checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_unet_no_roi/Checkpoint_best'
model.load_weights(checkpoint_path)
model.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',AUC(curve='PR'),
                            MeanAbsoluteError(),
                            MeanIoU(num_classes=2)])

model_roi=models.Unet(image_size=(img_size,img_size,1))
checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_unet_roi/Checkpoint_best'
model_roi.load_weights(checkpoint_path)
model_roi.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',AUC(curve='PR'),
                            MeanAbsoluteError(),
                            MeanIoU(num_classes=2)])

tm_unet=func.inference_time(model,model_roi,image_size=img_size,dim=1,itr=50)
print('U-Net time: {:.4f} ms'.format(np.mean(tm_unet)))
print(tm_unet)



########################################################################################################################
#img_size=448

# UNet
#model=models.Unetpp(image_size=(img_size,img_size,1))
#checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_unetpp/Checkpoint_best'
#model.load_weights(checkpoint_path)
#model.compile(optimizer=Adam(learning_rate=0.0001),
#                   loss='binary_crossentropy',
#                   metrics=['acc',AUC(curve='PR'),
#                            MeanAbsoluteError(),
#                            MeanIoU(num_classes=2)])

#model_roi=models.Unetpp(image_size=(img_size,img_size,1))
#checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_unetpp_roi/Checkpoint_best'
#model_roi.load_weights(checkpoint_path)
#model_roi.compile(optimizer=Adam(learning_rate=0.0001),
#                   loss='binary_crossentropy',
#                   metrics=['acc',AUC(curve='PR'),
#                            MeanAbsoluteError(),
#                            MeanIoU(num_classes=2)])

#tm_unet=func.inference_time(model,model_roi,image_size=img_size,dim=1,itr=50)
#print('U-Net time: {:.4f} ms'.format(np.mean(tm_unet)))
#print(tm_unet)



########################################################################################################################
#img_size=448

# UNet
#model=models.mobile_cellNet(image_size=(img_size,img_size,1))
#checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_mobile_cellNet_seg/Checkpoint_best'
#model.load_weights(checkpoint_path)
#model.compile(optimizer=Adam(learning_rate=0.0001),
#                   loss='binary_crossentropy',
#                   metrics=['acc',AUC(curve='PR'),
#                            MeanAbsoluteError(),
#                            MeanIoU(num_classes=2)])

#model_roi=models.mobile_cellNet(image_size=(img_size,img_size,1))
#checkpoint_path='new_test_set/results_patches/model_checkpoint_acc_mobile_cellNet_roi/Checkpoint_best'
#model_roi.load_weights(checkpoint_path)
#model_roi.compile(optimizer=Adam(learning_rate=0.0001),
#                   loss='binary_crossentropy',
#                   metrics=['acc',AUC(curve='PR'),
#                            MeanAbsoluteError(),
#                            MeanIoU(num_classes=2)])

#tm_unet=func.inference_time(model,model_roi,image_size=img_size,dim=1,itr=50)
#print('Mobile-CellNet time: {:.4f} ms'.format(np.mean(tm_unet)))
#print(tm_unet)