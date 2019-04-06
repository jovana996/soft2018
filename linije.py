import cv2
import numpy as np
from numpy.linalg import norm
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

def prosao_broj(linija, broj):
    x, y, w, h = broj
    centar = (int(x + (w / 2.0)), int(y + (h / 2.0)))

    t1, t2 = linija

    if t1[0] < centar[0] < t2[0] and  t2[1] < centar[1] < t1[1]:
        p1 = np.array(t1)
        p2 = np.array(t2)
        distance = norm(np.cross(p2 - p1, p1 - np.array(centar))) / norm(p2 - p1)

        if 15 > distance:
            return True

    return False
