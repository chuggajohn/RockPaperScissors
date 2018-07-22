import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

import cv2

cam = cv2.VideoCapture(0)

rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))

k = cv2.waitKey(1)
while k%256 != 27:
    k = cv2.waitKey(1)
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    
    
cv2.destroyAllWindows()

ret, oldframe = cam.read()
fgbg2 = cv2.createBackgroundSubtractorMOG2(0,50)
fgmask2 = fgbg2.apply(oldframe) 

 
while True:
    ret, frame = cam.read()
    fgmask2 = fgbg2.apply(frame)
    what = cv2.bitwise_and(frame, frame, mask=fgmask2)
    cv2.imshow("masked applied",what)
    fgbg2 = cv2.createBackgroundSubtractorMOG2(0,50)
    fgmask2 = fgbg2.apply(oldframe) 
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed, exit
        print("Escape hit, closing window...")
        break
    
cam.release()

cv2.destroyAllWindows()


