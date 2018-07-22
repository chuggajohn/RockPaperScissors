import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

import math

import cv2

def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

cam = cv2.VideoCapture(0)

rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))

k = cv2.waitKey(1)
while k%256 != 27:
    #take first frame
    k = cv2.waitKey(1)
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    
    
cv2.destroyAllWindows()

ret, oldframe = cam.read()
fgbg2 = cv2.createBackgroundSubtractorMOG2(0,50)
fgmask2 = fgbg2.apply(oldframe) 

 
while True:
    ret, frame = cam.read()
    fgmask2 = fgbg2.apply(frame, 0)
    what = cv2.bitwise_and(frame, frame, mask=fgmask2)
    cv2.imshow("masked applied",what)
    gs = cv2.cvtColor(what,cv2.COLOR_RGB2GRAY)
    #cv2.imshow("Gscale", gs)
    ret, Thold = cv2.threshold(gs, 31, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Threshold", Thold)
    Medianed = cv2.medianBlur(Thold, 9)
    #cv2.imshow("median", Medianed)
    closing = cv2.morphologyEx(Medianed, cv2.MORPH_CLOSE, rectkernel)
    #cv2.imshow("closing", closing)

    _,contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(what.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        isFinishCal,cnt = calculateFingers(res,drawing)
                    
        cv2.imshow('output', drawing)
        

     

    fgbg2 = cv2.createBackgroundSubtractorMOG2(0,50)
    fgmask2 = fgbg2.apply(oldframe)
    k = cv2.waitKey(10)
    if k%256 == 27:
        # ESC pressed, exit
        print("Escape hit, closing window...")
        break
    
cam.release()

cv2.destroyAllWindows()


