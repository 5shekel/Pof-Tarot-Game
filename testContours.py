"""
Card Recognition using OpenCV
docs here
"""

import sys
import numpy as np
import cv2

im = cv2.imread('test.jpg')

def getCards(im, numcards=4):
  imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(imgray,(1,1),1000)
  ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
  
  _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]  
  cv2.drawContours(im, contours, -1, (0,255,0), 2)
  cv2.imshow('source',im)
  cv2.waitKey(0)
  
  for card in contours:
    peri = cv2.arcLength(card,True)
    approx = rectify(cv2.approxPolyDP(card,0.02*peri,True))  
    h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
    
    transform = cv2.getPerspectiveTransform(approx,h)
    warp = cv2.warpPerspective(im,transform,(450,450))
    cv2.imshow('a',warp)      
    cv2.waitKey(0)

def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew
  
if __name__ == '__main__':
    cards = getCards(im,4)
    print cards
