import cv2
import numpy as np
import matplotlib.pyplot as plt
import linije
import math
import brojevi

xx1, yy1, xx2, yy2 = 0, 0, 0, 0

def ucitavanje_videa(video_path):
    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) # indeksiranje frejmova
    # analiza videa frejm po frejm
    while True:
        frame_num += 1
        ret_val, frame = cap.read()

        # ako frejm nije zahvacen
        if not ret_val:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresholdSlike, image_bin = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
        print("PRAG JE: " + str(thresholdSlike))

        (x1,y1),(x2, y2)  = linije.pronadji_liniju(frame)
        razdaljina = math.sqrt(((x1 - x2) ** 2) + (y1 - y2) ** 2)
        print("RAZDALJINA", razdaljina)
        if razdaljina > 150:
            xx1 = x1;
            yy1 = y1;
            xx2 = x2;
            yy2 = y2;
        else:
            continue

        linija = (xx1,yy1),(xx2, yy2)
        cv2.line(frame, linija[0], linija[1], [0,255,0], thickness=2)
        slika_sa_br = brojevi.izdvoj_brojeve(frame)
        plt.imshow(frame)
        plt.show()

def ucitaj_sve() :
     for i in [0,1,2, 3, 4, 5, 6, 7, 8, 9]:
        ucitavanje_videa('videos/video-' + str(i) + '.avi')

def main():
    print('cao zdravo')
    ucitaj_sve()

if  __name__ =="__main__" :
    main();