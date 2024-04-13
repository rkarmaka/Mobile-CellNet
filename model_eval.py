import sys
sys.path.insert(0, './utils')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, MeanAbsoluteError, MeanIoU
import numpy as np
import pandas as pd
from keras_flops import get_flops
import load_images, models, func
########################################################################################################################

img_size=(224,224,1)
X_test, y_test = load_images.load_images_test(img_size)
print(X_test.shape)
########################################################################################################################
# U-Net
print('###'*15)
print('Running U-Net Model')
print('###'*15)
model=models.Unet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_unet/Checkpoint_best'
model.load_weights(filepath=checkpoint_path)
model.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',AUC(curve='PR'),
                            func.dice_coef,
                            MeanAbsoluteError(),
                            MeanIoU(num_classes=2)])

# params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
#
# flops = np.round(get_flops(models.Unet(img_size),batch_size=1)/10**9,2)
# test_df=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
#              columns=['U-Net'],index=['params (M)','flops (G)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
#                                         'auc','dice','mae','iou','dc_2']).T


pred_test=model.predict(X_test)

cd=np.zeros(pred_test.shape[0])
cov=np.zeros(pred_test.shape[0])
num=np.zeros(pred_test.shape[0])
hx=np.zeros(pred_test.shape[0])

cd_pred=np.zeros(pred_test.shape[0])
cov_pred=np.zeros(pred_test.shape[0])
num_pred=np.zeros(pred_test.shape[0])
hx_pred=np.zeros(pred_test.shape[0])

for i in range(pred_test.shape[0]):
    cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]).astype(np.uint8),plot=False)
    cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)

cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
cd_df.to_csv('UNet_test.csv')
########################################################################################################################
# Residual UNet
print('###'*15)
print('Running Residual U-Net Model')
print('###'*15)
model=models.Res_Unet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_res_unet/Checkpoint_best'
model.load_weights(filepath=checkpoint_path)
model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

# params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
#
# flops = np.round(get_flops(models.Res_Unet(img_size),batch_size=1)/10**9,2)
# temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
#              columns=['Residual U-Net'],index=['params (M)','flops (G)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
#                                         'auc','dice','mae','iou','dc_2']).T

#test_df=pd.concat([test_df,temp])

################################
pred_test=model.predict(X_test)

cd=np.zeros(pred_test.shape[0])
cov=np.zeros(pred_test.shape[0])
num=np.zeros(pred_test.shape[0])
hx=np.zeros(pred_test.shape[0])

cd_pred=np.zeros(pred_test.shape[0])
cov_pred=np.zeros(pred_test.shape[0])
num_pred=np.zeros(pred_test.shape[0])
hx_pred=np.zeros(pred_test.shape[0])

for i in range(pred_test.shape[0]):
    cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]*255).astype(np.uint8),plot=False)
    cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)

cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
cd_df.to_csv('Res_UNet_test.csv')
########################################################################################################################
# UNet++
# print('###'*15)
# print('Running U-Net++ Model')
# print('###'*15)
# model=models.Unetpp(image_size=img_size)
# checkpoint_path='results_patches/model_checkpoint_acc_unetpp/Checkpoint_best'
# model.load_weights(filepath=checkpoint_path)
# model.compile(optimizer=Adam(learning_rate=0.0001),
#                        loss='binary_crossentropy',
#                        metrics=['acc',AUC(curve='PR'),
#                                 func.dice_coef,
#                                 MeanAbsoluteError(),
#                                 MeanIoU(num_classes=2)])

# params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
#
# flops = get_flops(models.Unetpp(img_size),batch_size=1)
# temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
#              columns=['U-Net++'],index=['params','flops(M)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
#                                         'auc','dice','mae','iou','dc_2']).T
#
# test_df=pd.concat([test_df,temp])

################################
# pred_test=model.predict(X_test)
#
# cd=np.zeros(pred_test.shape[0])
# cov=np.zeros(pred_test.shape[0])
# num=np.zeros(pred_test.shape[0])
# hx=np.zeros(pred_test.shape[0])
#
# cd_pred=np.zeros(pred_test.shape[0])
# cov_pred=np.zeros(pred_test.shape[0])
# num_pred=np.zeros(pred_test.shape[0])
# hx_pred=np.zeros(pred_test.shape[0])
#
# for i in range(pred_test.shape[0]):
#     cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]*255).astype(np.uint8),plot=False)
#     cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)
#
# cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
# cd_df.to_csv('UNetpp_test.csv')
########################################################################################################################
# Residual-UNet++
# print('###'*15)
# print('Running Residual U-Net++ Model')
# print('###'*15)
# model=models.Res_Unetpp(image_size=img_size)
# checkpoint_path='results_patches/model_checkpoint_acc_res_unetpp/Checkpoint_best'
# model.load_weights(filepath=checkpoint_path)
# model.compile(optimizer=Adam(learning_rate=0.0001),
#                        loss='binary_crossentropy',
#                        metrics=['acc',AUC(curve='PR'),
#                                 func.dice_coef,
#                                 MeanAbsoluteError(),
#                                 MeanIoU(num_classes=2)])
#
# # params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# # dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
# #
# # flops = get_flops(models.Res_Unetpp(img_size),batch_size=1)
# # temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
# #              columns=['Residual U-Net++'],index=['params','flops(M)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
# #                                         'auc','dice','mae','iou','dc_2']).T
# #
# # test_df=pd.concat([test_df,temp])
#
# ################################
# pred_test=model.predict(X_test)
#
# cd=np.zeros(pred_test.shape[0])
# cov=np.zeros(pred_test.shape[0])
# num=np.zeros(pred_test.shape[0])
# hx=np.zeros(pred_test.shape[0])
#
# cd_pred=np.zeros(pred_test.shape[0])
# cov_pred=np.zeros(pred_test.shape[0])
# num_pred=np.zeros(pred_test.shape[0])
# hx_pred=np.zeros(pred_test.shape[0])
#
# for i in range(pred_test.shape[0]):
#     cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]*255).astype(np.uint8),plot=False)
#     cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)
#
# cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
# cd_df.to_csv('Res_UNetpp_test.csv')
########################################################################################################################
# SegNet
print('###'*15)
print('Running SegNet Model')
print('###'*15)
model=models.SegNet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_segnet/Checkpoint_best'
model.load_weights(filepath=checkpoint_path)
model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

# params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
#
# flops = get_flops(models.SegNet(img_size),batch_size=1)
# temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
#              columns=['SegNet'],index=['params','flops(M)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
#                                         'auc','dice','mae','iou','dc_2']).T
#
# test_df=pd.concat([test_df,temp])

################################
pred_test=model.predict(X_test)

cd=np.zeros(pred_test.shape[0])
cov=np.zeros(pred_test.shape[0])
num=np.zeros(pred_test.shape[0])
hx=np.zeros(pred_test.shape[0])

cd_pred=np.zeros(pred_test.shape[0])
cov_pred=np.zeros(pred_test.shape[0])
num_pred=np.zeros(pred_test.shape[0])
hx_pred=np.zeros(pred_test.shape[0])

for i in range(pred_test.shape[0]):
    cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]*255).astype(np.uint8),plot=False)
    cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)

cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
cd_df.to_csv('SegNet_test.csv')
########################################################################################################################
# Mobile-CellNet - BCE Loss
print('###'*15)
print('Running Mobile-CellNet - BCE Loss Model')
print('###'*15)
model=models.mobile_cellNet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_mobile_cellNet_bce/Checkpoint_best'
model.load_weights(filepath=checkpoint_path)
model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

# params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
#
# flops = get_flops(models.mobile_cellNet(img_size),batch_size=1)
# temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
#              columns=['Mobile-CellNet-BCE'],index=['params','flops(M)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
#                                         'auc','dice','mae','iou','dc_2']).T
#
# test_df=pd.concat([test_df,temp])

################################
pred_test=model.predict(X_test)

cd=np.zeros(pred_test.shape[0])
cov=np.zeros(pred_test.shape[0])
num=np.zeros(pred_test.shape[0])
hx=np.zeros(pred_test.shape[0])

cd_pred=np.zeros(pred_test.shape[0])
cov_pred=np.zeros(pred_test.shape[0])
num_pred=np.zeros(pred_test.shape[0])
hx_pred=np.zeros(pred_test.shape[0])

for i in range(pred_test.shape[0]):
    cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]*255).astype(np.uint8),plot=False)
    cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)

cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
cd_df.to_csv('Mobile_CellNet_bce_test.csv')
########################################################################################################################
# Mobile-CellNet - Dice Loss
print('###'*15)
print('Running Mobile-CellNet - Dice Loss Model')
print('###'*15)
model=models.mobile_cellNet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_mobile_cellNet_dice/Checkpoint_best'
model.load_weights(filepath=checkpoint_path)
model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

# params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
#
# flops = get_flops(models.mobile_cellNet(img_size),batch_size=1)
# temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
#              columns=['Mobile-CellNet-Dice'],index=['params','flops(M)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
#                                         'auc','dice','mae','iou','dc_2']).T
#
# test_df=pd.concat([test_df,temp])

################################
pred_test=model.predict(X_test)

cd=np.zeros(pred_test.shape[0])
cov=np.zeros(pred_test.shape[0])
num=np.zeros(pred_test.shape[0])
hx=np.zeros(pred_test.shape[0])

cd_pred=np.zeros(pred_test.shape[0])
cov_pred=np.zeros(pred_test.shape[0])
num_pred=np.zeros(pred_test.shape[0])
hx_pred=np.zeros(pred_test.shape[0])

for i in range(pred_test.shape[0]):
    cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]*255).astype(np.uint8),plot=False)
    cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)

cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
cd_df.to_csv('Mobile_CellNet_dice_test.csv')
########################################################################################################################
# Mobile-CellNet - Jaccard Loss
print('###'*15)
print('Running Mobile-CellNet - Jaccard Loss Model')
print('###'*15)
model=models.mobile_cellNet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_mobile_cellNet_jac/Checkpoint_best'
model.load_weights(filepath=checkpoint_path)
model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

# params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
#
# flops = get_flops(models.mobile_cellNet(img_size),batch_size=1)
# temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
#              columns=['Mobile-CellNet-Jaccard'],index=['params','flops(M)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
#                                         'auc','dice','mae','iou','dc_2']).T
#
# test_df=pd.concat([test_df,temp])

################################
pred_test=model.predict(X_test)

cd=np.zeros(pred_test.shape[0])
cov=np.zeros(pred_test.shape[0])
num=np.zeros(pred_test.shape[0])
hx=np.zeros(pred_test.shape[0])

cd_pred=np.zeros(pred_test.shape[0])
cov_pred=np.zeros(pred_test.shape[0])
num_pred=np.zeros(pred_test.shape[0])
hx_pred=np.zeros(pred_test.shape[0])

for i in range(pred_test.shape[0]):
    cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]*255).astype(np.uint8),plot=False)
    cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)

cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
cd_df.to_csv('Mobile_CellNet_jac_test.csv')
########################################################################################################################
# Mobile-CellNet - Mixed Loss
print('###'*15)
print('Running Mobile-CellNet - Mixed Loss Model')
print('###'*15)
model=models.mobile_cellNet(image_size=img_size)
checkpoint_path='results_patches/model_checkpoint_acc_mobile_cellNet_mix/Checkpoint_best'
model.load_weights(filepath=checkpoint_path)
model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['acc',AUC(curve='PR'),
                                func.dice_coef,
                                MeanAbsoluteError(),
                                MeanIoU(num_classes=2)])

# params, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, \
# dice, mae, iou, dc_2 = func.test_performance(model,X_test,y_test)
#
# flops = get_flops(models.mobile_cellNet(img_size),batch_size=1)
# temp=pd.DataFrame([params, flops, sensitivity, specificity, accuracy, f2_score, dice_2, iou_2, jac_coeff, auc, dice, mae, iou, dc_2],
#              columns=['Mobile-CellNet-Mixed'],index=['params','flops(M)','sensitivity','specificity','accuracy','f2_score','dice_2','iou_2','jac_coedd',
#                                         'auc','dice','mae','iou','dc_2']).T
#
# test_df=pd.concat([test_df,temp])

################################
pred_test=model.predict(X_test)

cd=np.zeros(pred_test.shape[0])
cov=np.zeros(pred_test.shape[0])
num=np.zeros(pred_test.shape[0])
hx=np.zeros(pred_test.shape[0])

cd_pred=np.zeros(pred_test.shape[0])
cov_pred=np.zeros(pred_test.shape[0])
num_pred=np.zeros(pred_test.shape[0])
hx_pred=np.zeros(pred_test.shape[0])

for i in range(pred_test.shape[0]):
    cd[i],cov[i],num[i],hx[i]=func.cell_density((y_test[i]*255).astype(np.uint8),plot=False)
    cd_pred[i],cov_pred[i],num_pred[i],hx_pred[i]=func.cell_density((pred_test[i]*255).astype(np.uint8),plot=False)

cd_df=pd.DataFrame([cd,cd_pred,cov,cov_pred,num,num_pred,hx,hx_pred],index=['CD','CD_pred','CV','CV_pred','NUM','NUM_pred','HEX','HEX_pred']).T
cd_df.to_csv('Mobile_CellNet_mix_test.csv')


#test_df.to_csv('Test_results.csv')