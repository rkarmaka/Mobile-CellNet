import cv2 as cv
import numpy as np
from skimage.morphology import convex_hull_image, diameter_closing
import matplotlib.pyplot as plt

#### imclearborder definition
def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv.findContours(imgBWcopy.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv.findContours(imgBWcopy.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

def hexa(im,plot,pred):
    if pred:
        cont,h=cv.findContours(cv.GaussianBlur(im,(3,3),0.2),cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        approx = [cv.approxPolyDP(cont[i],0.03 * cv.arcLength(cont[3], True), True) for i in range(len(cont)) if ((cv.contourArea(cont[i])>10)&
                                                                                                               (cv.contourArea(cont[i])<2000))]
        
        M=[cv.moments(cont[i]) for i in range(len(cont)) if ((cv.contourArea(cont[i])>10)&(cv.contourArea(cont[i])<2000))]
        if plot:
            plt.figure(figsize=(6,6))
            plt.imshow(im,cmap='gray')
            for j in range(len(approx)):
                cx = int(M[j]['m10']/M[j]['m00'])
                cy = int(M[j]['m01']/M[j]['m00'])
                x=[approx[j][i][0][0] for i in range(len(approx[j]))]
                y=[approx[j][i][0][1] for i in range(len(approx[j]))]
                plt.scatter(x,y,c='red',marker='.')
                plt.scatter(cx,cy,c='red',marker='*')
        
        #hull_list = [cv.convexHull(cont[i]) for i in range(len(cont)) if ((cv.contourArea(cont[i])>50)&
        #                                                                (cv.contourArea(cont[i])<2000))]
        
        hull_list=convex_hull_image(im)
        #plt.figure()
        #plt.imshow(hull_list)
        #h=np.array([len(hull_list[i]) for i in range(len(hull_list))])
        #print(h)
        #drawing = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
        #for i in range(len(cont)):
        #    color = (randint(0,256), randint(0,256), randint(0,256))
        #    cv.drawContours(drawing, hull_list, i, color)
                
        h=np.array([len(approx[i]) for i in range(len(approx))])
        #print(h)
        
    else:
        cont,h=cv.findContours(cv.GaussianBlur(im,(3,3),0.1),cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        approx = [cv.approxPolyDP(cont[i],0.025 * cv.arcLength(cont[3], True), True) for i in range(len(cont))]
        if plot:
            plt.figure(figsize=(6,6))
            plt.imshow(im,cmap='gray')
        for j in range(len(approx)):
            x=[approx[j][i][0][0] for i in range(len(approx[j]))]
            y=[approx[j][i][0][1] for i in range(len(approx[j]))]
            if plot:
                plt.scatter(x,y,c='red',marker='.')
        h=np.array([len(approx[i]) for i in range(len(approx))])
    return np.sum(h==6)*100/len(h)

def cell_density(img, out_file,plot=True,pred=False):
    if pred:
        _,imbw=cv.threshold(img,200,img.max(),cv.THRESH_BINARY)
        imbw=imclearborder(imbw,1)
        k=se_hexa((9,11))
        imbw=cv.morphologyEx(imbw,cv.MORPH_OPEN,kernel=k.astype(np.uint8))
        imbw=diameter_closing((255-imbw), diameter_threshold=16)
        imbw=(imbw-255)*255

        cv.imwrite(out_file, imbw)

    else:
        _,imbw=cv.threshold(img,img.max()/2,img.max(),cv.THRESH_BINARY)
        imbw=imclearborder(imbw,1)

    cont,_ =cv.findContours(imbw,cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    num=len(cont)
    if num>10:
        if pred:
            area=[cv.contourArea(cont[i]) for i in range(len(cont)) if ((cv.contourArea(cont[i])>50)&
                                                                        (cv.contourArea(cont[i])<2000))]
            num=len(area)
            cd=np.round((num*446*304*10*255)/(np.sum(imbw)*1.1),0)
            cov=np.round(np.std(area)*100/np.mean(area),0)
            #hex_=np.round(hexa(imbw,plot,pred),0)
            hex_=np.round(hexa_circ(imbw,plot=plot,pred=pred),0)
        else:
            cd=np.round((num*446*304*10)/(np.sum(imbw)*1.05),0)
            area=[cv.contourArea(cont[i]) for i in range(len(cont))]
            cov=np.round(np.std(area)*100/np.mean(area),0)
            hex_=np.round(hexa(imbw,plot,pred),0)
    else:
        cd=0
        cov=0
        hex_=0
    return cd,cov,num,hex_

def postProc_roi(img,pred=True):
    if pred:
        _,imbw=cv.threshold(img,img.max()/2,img.max(),cv.THRESH_BINARY)
        imbw=cv.morphologyEx(imbw,cv.MORPH_OPEN,kernel=np.ones((15,15),np.uint8))

        contours, hierarchy = cv.findContours((255*imbw).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        area=[cv.contourArea(contours[i]) for i in range(len(contours))]
        new_imbw=cv.drawContours(np.ones((446,304)),contours,area.index(max(area)),(0,0,255),thickness=cv.FILLED)
        
    else:
        _,imbw=cv.threshold(img,img.max()/2,img.max(),cv.THRESH_BINARY)
        imbw=cv.morphologyEx(imbw,cv.MORPH_CLOSE,kernel=np.ones((15,15),np.uint8))

        contours, hierarchy = cv.findContours(imbw.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            new_imbw=np.ones((446,304))
        else:
            area=[cv.contourArea(contours[i]) for i in range(len(contours))]
            new_imbw=cv.drawContours(np.ones((446,304)),contours,area.index(max(area)),(0,0,255),thickness=cv.FILLED)

    return 1-new_imbw

def se_hexa(shape):
    k=np.ones(shape)
    for i in range(np.shape(k)[0]//2):
        for j in range(0,np.shape(k)[0]//2-i):
            k[i,j]=0
            k[i,-j-1]=0
            k[-i-1,j]=0
            k[-i-1,-j-1]=0
    return k



def hexa_circ(im,pred=False,plot=False):
    if pred:
        cont,h=cv.findContours(cv.GaussianBlur(im,(3,3),0.2),cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        area = [cv.contourArea(cont[i]) for i in range(len(cont)) if ((cv.contourArea(cont[i])>50)&(cv.contourArea(cont[i])<2000))]
        perim = [cv.arcLength(cont[i], True) for i in range(len(cont)) if ((cv.contourArea(cont[i])>50)&(cv.contourArea(cont[i])<2000))]
        h = [4*np.pi*area[i]/perim[i]**2 for i in range(len(area))]
        h2 = [h[i] for i in range(len(h)) if ((h[i]>0.86) & (h[i]<0.91))]
        hxx = len(h2)*100/len(h)
        
        approx = [cv.approxPolyDP(cont[i],0.03 * cv.arcLength(cont[3], True), True) for i in range(len(cont)) if ((cv.contourArea(cont[i])>50)&
                                                                                                               (cv.contourArea(cont[i])<2000))]
        
    M=[cv.moments(cont[i]) for i in range(len(cont)) if ((cv.contourArea(cont[i])>50)&(cv.contourArea(cont[i])<2000))]
    if plot:
        plt.figure(figsize=(6,6))
        plt.imshow(im,cmap='binary_r')
        for j in range(len(approx)):
            cx = int(M[j]['m10']/M[j]['m00'])
            cy = int(M[j]['m01']/M[j]['m00'])
            #x=[approx[j][i][0][0] for i in range(len(approx[j]))]
            #y=[approx[j][i][0][1] for i in range(len(approx[j]))]
            #plt.scatter(x,y,c='red',marker='.')
            plt.scatter(cx,cy,c='red',marker='.')
            plt.axis('off')
        
    else:
        cont,h=cv.findContours(cv.GaussianBlur(im,(3,3),0.1),cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        approx = [cv.approxPolyDP(cont[i],0.025 * cv.arcLength(cont[3], True), True) for i in range(len(cont))]
        if plot:
            plt.figure(figsize=(6,6))
            plt.imshow(im,cmap='gray')
        for j in range(len(approx)):
            x=[approx[j][i][0][0] for i in range(len(approx[j]))]
            y=[approx[j][i][0][1] for i in range(len(approx[j]))]
            if plot:
                plt.scatter(x,y,c='red',marker='.')
        h=np.array([len(approx[i]) for i in range(len(approx))])
        hxx=np.sum(h==6)*100/len(h)
    return hxx