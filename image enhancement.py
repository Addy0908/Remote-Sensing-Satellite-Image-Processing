
import os
import numpy as np
import cv2
import argparse



def add( matrix1: np, matrix2: np) -> np:
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if int(matrix1[i][j]) + int(matrix2[i][j]) > 255:
                matrix1[i][j] = 255
            else:
                matrix1[i][j] = int(matrix1[i][j]) + int(matrix2[i][j])
    return matrix1


imageList = os.listdir('testImage')

for i in imageList:
    
    img = cv2.imread('testImage/'+i)
    img = cv2.resize(img, (500, 500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    # Radiometric Enhancement of Image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgY = clahe.apply(img)
    cv2.imwrite('radiometricEnhancement_1/'+i, imgY)
    
    
    # Spatial Enhancement of Image
    imgX = cv2.bilateralFilter(img,9,80,80)
    
    
    img2 = cv2.Laplacian(imgX,cv2.CV_64F)
    img2 = add(img, img2) 
    
    img = cv2.Sobel(img2,cv2.CV_64F,dx= 2,dy =2,ksize=3)
    img3 = cv2.GaussianBlur(img,(5,5),0)
    
    img = add(img,img3)

    img = np.array(255*(np.abs(img/255)**1.3),dtype='uint8')
    img = add(img,imgY)
    imgNew = img
    cv2.imwrite('spatialEnhancement_2/'+i, img)

    # Spectral Enhancement of Image
    img[:,:] = cv2.equalizeHist(img[:,:])
    # img = add(imgNew, img)
    cv2.imwrite('spectralEnhancement_3/'+i, img)
    

    # cv2.imwrite('geometricEnhancement_4/'+i, img)
