import cv2
import numpy as np
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)

#cv2.xfeatures2d.SIFT_create()
fgbg = cv2.createBackgroundSubtractorMOG2()

ret1, frame1 = cap.read();

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame, frame1)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


