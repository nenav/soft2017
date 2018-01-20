import numpy as np
import cv2



def load_digits_image(filepath):
    print("Ucitavanje slike: [" + filepath + "] ...")
    digits_image = cv2.imread(filepath, 0)
    cv2.imshow("Obucavajuca Slika", digits_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
