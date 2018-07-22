import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
#cv2.namedWindow("masked img")

img_counter = 0

rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))

fgbg2 = cv2.createBackgroundSubtractorMOG2()
fgbg2.setVarThreshold(120)
img_bg = "opencv_frame_0.png"
img = "opencv_frame_1.png"
img_preset = "opencv_frame_0.png"
changed = False;
bg = cv2.imread(img_preset)
fgmask2 = fgbg2.apply( bg, learningRate=0.00)

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed, exit
        print("Escape hit, closing window...")
        break

    elif k%256 == ord('a'):
        # a pressed, press twice, first for background second for foreground
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        if(img_counter > 1):
            img_counter = 0
            changed = True

    elif k%256 == 32:
        # SPACE pressed, calculate the foreground objects
        if changed == True:
            bg = cv2.imread(img_bg)
            fgbg2 = cv2.createBackgroundSubtractorMOG2()
            fgmask2 = fgbg2.apply( bg, learningRate=0.00)
            img_new = img
        else:
            img_new = img

        print "Opening foreground ", changed, " ", img_new
        newimg = cv2.imread(img_new)
        fgmask2 = fgbg2.apply( newimg, learningRate=0)
        cv2.imshow("masked img2 applied", fgmask2)

        fgmask2 = cv2.medianBlur(fgmask2,11)
        #cv2.imshow("masked img2 after blur", fgmask2)

        ret, fgmask2 = cv2.threshold(fgmask2, 30, 250, cv2.THRESH_BINARY)
        #cv2.imshow("masked img2 before op,close", fgmask2)

        opening = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, rectkernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, rectkernel)
        #cv2.imshow("masked img2 after closing", closing)
        imgcon, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        image_of_contours = cv2.drawContours(imgcon, contours, -1, (55,255,0),3)
        cv2.imshow("masked img2", image_of_contours)
cam.release()

cv2.destroyAllWindows()


