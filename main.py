import cv2
import numpy as np
import matplotlib.pyplot as plt
import linije
import math
import brojevi
import test

import neuronska_mreza as nm


xx1, yy1, xx2, yy2 = 0, 0, 0, 0

def ucitavanje_videa(video_path, ann):
    # ucitavanje videa
    frame_num = 0
    rezultat = 0;
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


        (x1, y1), (x2, y2) = linije.pronadji_liniju(frame)
        razdaljina = math.sqrt(((x1 - x2) ** 2) + (y1 - y2) ** 2)

        if razdaljina > 220:
            xx1 = x1;
            yy1 = y1;
            xx2 = x2;
            yy2 = y2;
        else:
            continue

        linija = (xx1, yy1), (xx2, yy2)
        cv2.line(frame, linija[0], linija[1], [0,255,0], thickness=2)
        contures_numbers = brojevi.izdvoj_brojeve(frame)
        #print("KONTURE SVIH BROJEVA : ", contures_numbers)
        #print("LINIJA : ", linija)
        for kontura in contures_numbers:
            if(linije.prosao_broj(linija, kontura)):
                print('PROSAO BROJ!!!!!!!')
                broj = brojevi.skaliranje_broja(frame,kontura)
                #plt.imshow(broj)
                #plt.show()
                broj = nm.matrix_to_vector(broj)
                izlaz = ann.predict_classes(broj.reshape(1, 28, 28, 1))


                #izlaz = ann.predict(broj)

                #print('IZLAZ : ', int(izlaz))
                rezultat += izlaz
        #plt.imshow(frame)
        #plt.show()
    return int(rezultat)
def ucitaj_sve(ann) :
     for i in [3, 4, 5, 6, 7, 8, 9]:
        ucitavanje_videa('videos/video-' + str(i) + '.avi', ann)

def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)
def main():

    klasifikator = nm.create_model()
    klasifikator.load_weights('weights.h5')
    #ucitaj_sve(klasifikator)


    all_copies = []

    with open('out.txt', 'w') as file:
        file.write('RA 61/2015 Jovana Novakovic\n')
        file.write('Video\tsuma\t\n')

    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print('obradjujem video..... : ', i)
        rezultat = ucitavanje_videa('videos/video-' + str(i) + '.avi', klasifikator)
        print('rezulat ', rezultat)
        with open('out.txt', 'a') as file:
            file.write('video-' + str(i) + '\t' + str(rezultat) + '\n')


    test.test()


if  __name__ =="__main__" :
    main()