import sys
sys.path.insert(0, './utils')
from tensorflow.keras.metrics import AUC, MeanAbsoluteError, MeanIoU
import numpy as np
import pandas as pd
from keras_flops import get_flops
import load_images, modules.models as models
########################################################################################################################

img_size=(224,224,1)
X_test, y_test = load_images.load_images_test(img_size)

########################################################################################################################
# U-Net
model_unet=models.Unet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_unet/Checkpoint_best'
model_unet.summary()
model_unet.load_weights(filepath=checkpoint_path)
model_unet.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',AUC(curve='PR'),
                            func.dice_coef,
                            MeanAbsoluteError(),
                            MeanIoU(num_classes=2)])

params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
dice, mae, iou, dc_2 = func.test_performance(model_unet,X_test,y_test)

flops = np.round(get_flops(models.Unet(img_size),batch_size=1)/10**9,2)
test_df=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
             columns=['U-Net'],index=['params (M)','flops (G)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
                                        'auc','dice','mae','iou','dc_2']).T

########################################################################################################################
# Residual UNet
model_res_unet=models.Res_Unet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_res_unet/Checkpoint_best'
model_res_unet.load_weights(filepath=checkpoint_path)
model_res_unet.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
dice, mae, iou, dc_2 = func.test_performance(model_res_unet,X_test,y_test)

flops = np.round(get_flops(models.Res_Unet(img_size),batch_size=1)/10**9,2)
temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
             columns=['Residual U-Net'],index=['params (M)','flops (G)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
                                        'auc','dice','mae','iou','dc_2']).T

test_df=pd.concat(test_df,temp)

########################################################################################################################
# UNet++
model_unetpp=models.Unetpp(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_unetpp/Checkpoint_best'
model_unetpp.load_weights(filepath=checkpoint_path)
model_unetpp.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
dice, mae, iou, dc_2 = func.test_performance(model_unetpp,X_test,y_test)

flops = get_flops(models.Unetpp(img_size),batch_size=1)
temp=pd.DataFrame([params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
             columns=['U-Net++'],index=['params','flops(M)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
                                        'auc','dice','mae','iou','dc_2']).T

test_df=pd.concat(test_df,temp)

########################################################################################################################
# Residual-UNet++
model_res_unetpp=models.Res_Unetpp(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_res_unetpp/Checkpoint_best'
model_res_unetpp.load_weights(filepath=checkpoint_path)
model_res_unetpp.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
dice, mae, iou, dc_2 = func.test_performance(model_unetpp,X_test,y_test)

temp=pd.DataFrame([sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
             columns=['Residual U-Net++'],index=['sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
                                        'auc','dice','mae','iou','dc_2']).T

test_df=pd.concat(test_df,temp)

########################################################################################################################
# SegNet
model_segnet=models.SegNet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_segnet/Checkpoint_best'
model_segnet.load_weights(filepath=checkpoint_path)
model_segnet.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
dice, mae, iou, dc_2 = func.test_performance(model_unetpp,X_test,y_test)

temp=pd.DataFrame([params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
             columns=['SegNet'],index=['params','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
                                        'auc','dice','mae','iou','dc_2']).T

test_df=pd.concat(test_df,temp)