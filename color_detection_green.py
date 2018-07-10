# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:13:21 2018

@author: Ayushi
"""

import cv2
import numpy as np
#import sqlite3

lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])                          #33degree-102 degreee hue contains green color
kernelOpen=np.ones((5,5))           #to remove extra noise which is smaller than this size
kernelClose=np.ones((20,20))

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img=cv2.resize(img,(340,220))
    
    #Convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #create the mask
    mask = cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology - noise removal
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    maskFinal = maskClose
    
    #to draw boundaries in the detected area
    _,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    #to draw a rectangle
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
    cv2.imshow('maskOpen',maskOpen)
    cv2.imshow('maskClose',maskClose)
    
    cv2.imshow('mask',mask)
    cv2.imshow('cam',img)
    cv2.waitKey(10)
    
cap.release()
cv2.destroyAllWindows()

    