import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def pronadji_liniju(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(frame,50,150,apertureSize = 3)



 #10 - minimum length of line
 #maxLineGap - Maximum allowed gap between line segments to treat them as single line.

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, maxLineGap=10)
    if lines[0] is not None:
     for x1, y1, x2, y2 in lines[0]:

        print("x1, y1 : ", x1, y1, " x2, y2 : ", x2, y2)
     return (x1, y1), (x2, y2)

