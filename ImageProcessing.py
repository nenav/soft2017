import numpy as np
import cv2

VELICINA_CIFRE = 20

def split_digits_image(digits_image, split_size):
    h, w = digits_image.shape[:2]
    print("Visina i sirina slike: " + str(h) + ", " + str(w))
    x, y = split_size
    print("Visina i sirina delova: " + str(y) + ", " + str(x))

    # np.vsplit(matrica, visina)
    # razdeli sliku po odredjenoj dimenziji
    # ovde se originalna slika deli po h//y -> // zaokruzi broj
    # ukupno => 1000 // 20 => 100 / 2 => 50 => 5 redova po svakoj cifri

    # np.hsplit(matrica, sirina)
    # isto kao i vsplit ali po sirini deli

    cells = []
    vsplitres = np.vsplit(digits_image, h // y)
    for row in vsplitres:
        hsplitres = np.hsplit(row, w//x)
        for col in hsplitres:
            cells.append(col)
    print("Ukupno ucitanih slika: " + str(len(cells)))

def load_digits_image(filepath):
    print("Ucitavanje slike: [" + filepath + "] ...")
    digits_image = cv2.imread(filepath, 0)
    split_digits_image(digits_image, (VELICINA_CIFRE, VELICINA_CIFRE))

