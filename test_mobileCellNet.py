from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, MeanAbsoluteError, MeanIoU



from modules.utils import load_test_data, dice_coef, read_image_file_names, pad_image
from modules.models import mobile_cellNet
from modules.processing import postProc_roi, cell_density


import cv2 as cv
import numpy as np




# load the dataset
root_folder = "/Users/ranit/Research/Mobile-CellNet"
# out_folder = "/Users/ranit/Research/Mobile-CellNet/output"


img_size = (448,448,1)
model_unet = mobile_cellNet(img_size)
print(model_unet.summary())
checkpoint_path = f'{root_folder}/model_checkpoints/model_checkpoint_acc_mobile_cellNet_seg/Checkpoint_best'
print(checkpoint_path)
model_unet.load_weights(checkpoint_path)
model_unet.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',AUC(curve='PR'),
                            dice_coef,
                            MeanAbsoluteError(),
                            MeanIoU(num_classes=2)])



model_unet_roi = mobile_cellNet(img_size)
checkpoint_path = f'{root_folder}/model_checkpoints/model_checkpoint_acc_mobile_cellNet_roi/Checkpoint_best'
model_unet_roi.load_weights(checkpoint_path)
model_unet_roi.compile(optimizer=Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['acc',AUC(curve='PR'),
                            dice_coef,
                            MeanAbsoluteError(),
                            MeanIoU(num_classes=2)])







image_files = read_image_file_names(root_folder=f'{root_folder}/images')



cd,cov,num,hex_ = [], [], [], []

for file in image_files:
    image_name = file.split('/')[-1].split('.')[0]
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    h, w = image.shape
    padded_image = pad_image(image, out_dim=(448,448))
    
    temp_cell=model_unet.predict(padded_image[np.newaxis,...])[0,:h,:w,0]
    temp_roi=postProc_roi(model_unet_roi.predict(padded_image[np.newaxis,...])[0,:h,:w])
    cd_temp,cov_temp,num_temp,hex_temp = cell_density((temp_cell*temp_roi*255).astype(np.uint8),
                                                      pred=True,plot=False, out_file=f'{root_folder}/output/{image_name}')
    
    cd.append(cd_temp)
    cov.append(cov_temp)
    num.append(num_temp)
    hex_.append(hex_temp)



# im_dim = (446,304)  # Original image dimension
# out_dim = (448,448)

# data_x = load_test_data(x_path, im_dim=im_dim, out_dim=out_dim)
# # pre-process X
# data_x = data_x/255.

# # data_y=load_test_data(y_path, im_dim=im_dim, out_dim=out_dim)
# # # pre-process Y
# # _,data_y=cv.threshold(data_y,200,255,cv.THRESH_BINARY)
# # data_y=data_y/255.








# n_image = len(image_files)










# idx=3
# temp1=model_unet.predict(data_x[np.newaxis,idx,...])[0,:446,:304,0]
# temp2=postProc_roi(model_unet_roi.predict(data_x[np.newaxis,idx,...])[0,:446,:304])
# img=(temp1*temp2*255).astype(np.uint8)
# _,imbw=cv.threshold(img,200,img.max(),cv.THRESH_BINARY)

# imbw=imclearborder(imbw,1)
# imbw_1=imbw
# k=se_hexa((9,11))
# imbw=cv.morphologyEx(imbw,cv.MORPH_OPEN,kernel=k.astype(np.uint8))
# imbw=diameter_closing((255-imbw), diameter_threshold=16)
# imbw=(imbw-255)*255
# cont,h=cv.findContours(imbw,cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# num=len(cont)
# if num>10:
#     area=[cv.contourArea(cont[i]) for i in range(len(cont)) if ((cv.contourArea(cont[i])>50)&(cv.contourArea(cont[i])<2000))]
#     print("NUM: {}".format(len(area)))
#     print("CD: {}".format(np.round((num*446*304*10*255)/(np.sum(imbw)*1.1),0)))
#     print("CV: {}".format(np.round(np.std(area)*100/np.mean(area),0)))
#     print("HEX: {}".format(np.round(hexa_circ(imbw,plot=False,pred=True),0)))
        
               
# plt.imshow(imbw,cmap='gray')