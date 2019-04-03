import cv2
import numpy as np
import matplotlib.pyplot as plt



def izdvoj_brojeve(image):
    upper = np.array([255, 255, 255])
    lower = np.array([150, 150, 150])  # Posto su brojevi beli

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask) # na slici ostaje samo ono sto je jako svetlo (ovde - beli brojeavi)
    plt.imshow(output)
    plt.show()
    detekcija_brojeva(output)

    return output


def detekcija_brojeva(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gs = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    retSlike, image_bin = cv2.threshold(img_gs, 100, 255, cv2.THRESH_OTSU)
    print("Prag je: " + str(retSlike))

    plt.imshow(image_bin)
    plt.show()
    contours, hierarchy = cv2.findContours(img_gs, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = image.copy()
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    contours_numbers = []  # ovde ce biti samo konture koje pripadaju bar-kodu

    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect( contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size

        x, y, w, h = cv2.boundingRect(contour)

        if w > 3 and h > 5 and h < 300:  # uslov da kontura pripada (trebalo bi preko Krugova)
            contours_numbers.append(contour)  # ova kontura pripada bar-kodu
            print('Detektovano kontura(linija):  ' + str(len(contours_numbers)))
            print('Kordinate duzi su: ' + str(x) + ',' + str(y+h) + '  A druge tacke: ' + str(x+w) +',' + str(y)+ '  //Sirina je: ' + str(w) + ' __VISINA JE:  ' + str(h) )

            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            plt.imshow(img)
            plt.show()
        #im = cv2.drawContours(im, contours_numbers, 0, (0, 255, 0), 2)