import cv2
import numpy as np
import os
import random

path = r'/home/jap01/PycharmProjects/face recogiition original/database1'

def brightness(img, low, high):
    value = random.randint(low, high)/100
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img



for i in os.listdir(path):
    print(i)
    img = cv2.imread(path + '/' + i)
    cv2.imshow("frame",img)
    img = brightness(img,100,240)
    cv2.imshow("frame1", img)
    cv2.waitKey(0)
