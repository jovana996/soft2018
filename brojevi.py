import cv2
import numpy as np
import matplotlib.pyplot as plt



def izdvoj_brojeve(image):
    upper = np.array([255, 255, 255])
    lower = np.array([150, 150, 150])  # Posto su brojevi beli

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask) # na slici ostaje samo ono sto je jako svetlo (ovde - beli brojevi)
    #plt.imshow(output)
    #plt.show()
    contures_numbers = detekcija_brojeva(output)

    return contures_numbers


def detekcija_brojeva(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gs = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    retSlike, image_bin = cv2.threshold(img_gs, 100, 255, cv2.THRESH_OTSU)
    #print("Prag je: " + str(retSlike))

    #plt.imshow(image_bin)
    #plt.show()
    _, contours, hierarchy = cv2.findContours(img_gs, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = image.copy()
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    contours_numbers = []  # ovde ce biti samo konture koje su brojevi
    koordinate_brojeva = []

    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect( contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size

        x, y, w, h = cv2.boundingRect(contour)


        if w > 8 and h > 15 and h < 150:  # uslov da kontura pripada
            contours_numbers.append(contour)

            #print('Kordinate duzi su: ' + str(x) + ',' + str(y+h) + '  A druge tacke: ' + str(x+w) +',' + str(y)+ '  //Sirina je: ' + str(w) + ' __VISINA JE:  ' + str(h) )

            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            koordinate_brojeva.append((x,y,w,h))


      #      plt.imshow(img)
     #       plt.show()
    #print('Detektovano kontura(linija):  ' + str(len(contours_numbers)))
    return koordinate_brojeva
def skaliranje_broja(img, kontura):
    x,y,w,h = kontura
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    ret,img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    isecenBroj = img_bin[y:y + h, x:x + w]
    isecenBroj = cv2.resize(isecenBroj, (28, 28), interpolation=cv2.INTER_NEAREST)
    # radimo zatvaranje da popunimo 'crne rupe' u brojevima
    isecenBroj = cv2.dilate(isecenBroj, kernel, iterations=1)
    isecenBroj = cv2.erode(isecenBroj, kernel, iterations=1)
    isecenBroj = ispravi_broj(isecenBroj)
    cv2.imwrite('slikeBrojeva/brojevi.png', isecenBroj)

    return isecenBroj

def ispravi_broj(img):

    moments = cv2.moments(img)
    if abs(moments['mu02']) < 1e-2:
        return img.copy()
    nakrivljenost = moments['mu11'] / moments['mu02']
    M = np.float32([[1, nakrivljenost, -0.5 * 28 * nakrivljenost], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255


def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()
