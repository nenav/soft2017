import numpy as np
from numpy.linalg import norm
import cv2

VELICINA_CIFRE = 20
BROJ_KLASA = 10

knn_model = cv2.ml.KNearest_create()

def split_digits_image(digits_image, split_size):
    h, w = digits_image.shape[:2]
    print("Visina i sirina slike: " + str(h) + ", " + str(w))
    x, y = split_size
    print("Visina i sirina delova: " + str(y) + ", " + str(x))
    #TODO: np.vsplit i np.hsplit
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

    return cells

def deskew(img):
    #TODO:
    # Prepravljanje slike ako je ona nakosena preko momenata slike
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*VELICINA_CIFRE*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (VELICINA_CIFRE, VELICINA_CIFRE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def load_digits_image(filepath):
    print("Ucitavanje slike: [" + filepath + "] ...")
    digits_image = cv2.imread(filepath, 0)
    digits = split_digits_image(digits_image, (VELICINA_CIFRE, VELICINA_CIFRE))
    #TODO: np.reapeat i np.arange
    # np.arange -> rasporedi podjednako cifre od 0
    # np.arange(10) -> 0,1,2,3,4,5,6,7,8,9
    # np.repeat -> ponavlja pojavljivanje elemenata
    # np.repeat(10,2) -> 10 10
    # np.repeat([1,2,3], 2) -> [1,1,2,2,3,3]
    # np.repeat([0,1,2,3,4], 4) -> [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
    # zgodno koristiti ove dve funkcije za generisanje labela za trening skup:
    # np.repeat([0,1,2,3,4,5,6,7,8,9], 10) ->
    #   [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2 .....]
    labele = np.repeat(np.arange(BROJ_KLASA), len(digits) / BROJ_KLASA)

    return digits, labele

def preprocess(digit_images):
    samples = []
    for image in digit_images:
        thresholded_digit = np.zeros(image.shape, image.dtype)
        cv2.threshold(image, 255,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY, thresholded_digit)
        samples.append(thresholded_digit)
    return samples

def flatten_images(digit_images_thresholded):
    sample_image = np.zeros([len(digit_images_thresholded),VELICINA_CIFRE*VELICINA_CIFRE],dtype=np.float32)
    for img_idx in range(len(digit_images_thresholded)):
        flat_image = digit_images_thresholded[img_idx].reshape(1, VELICINA_CIFRE * VELICINA_CIFRE)
        sample_image[img_idx] = flat_image
    return sample_image

def flatten_image(image):
    flat_image = image.reshape(1, VELICINA_CIFRE * VELICINA_CIFRE)
    return flat_image

def test_knn(image_name):
    image = cv2.imread("assets/img_log/straigth/" + image_name, 0)
    thresholded_image = np.empty(image.shape, image.dtype)
    cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY, thresholded_image)
    flat_image = flatten_image(thresholded_image)
    flat_image = np.float32(flat_image)

    ret, res, neig, dist = knn_model.findNearest(flat_image, 5)
    print(ret,res,neig,dist)


def create_KNN(filepath):
    try:
        print('Pokusaj ucitavanja gotov skupa za trening:')
        samples_image = cv2.imread("assets/img_log/training-set/training-set.png", 0)
        samples_image = np.float32(samples_image)
        labels = np.repeat(np.arange(BROJ_KLASA), 5000 / BROJ_KLASA)
        print('Uspesno ucitavanje slike')
    except:
        print('Nije pronadjena slika sa trening podacima...')
        print('Kreiranje nove ... ')
        [digits, labels] = load_digits_image(filepath)
        digits_straight = list(map(deskew, digits))
        digits_thresholded = preprocess(digits_straight)
        samples_image = flatten_images(digits_thresholded)
        cv2.imwrite("assets/img_log/training-set/training-set.png", samples_image)

    knn_model.train(samples_image, cv2.ml.ROW_SAMPLE, labels)
    #TODO:
    #provera da li knn radi kako treba
    #test_knn("digit-straight-3095.png")

    #TODO:
    # upis svih dobijenih slika
    #for imgidx in range(len(digits)):
    #   cv2.imwrite("assets/img_log/skewed/digit-" + str(imgidx) + ".png", digits[imgidx])

    #TODO:
    # upis svih dobijenih prepravljenih slika
    #for imgidx in range(len(digits_straight)):
    #    cv2.imwrite("assets/img_log/straigth/digit-straight-" + str(imgidx)+ ".png", digits_straight[imgidx])

    #TODO:
    # upis svih dobijenih thresholdovanih slika
    #for imgidx in range(len(digits_thresholded)):
    #   cv2.imwrite("assets/img_log/thresholded/digit-straight-thresholded" + str(imgidx)+ ".png", digits_thresholded[imgidx])

    #TODO:
    #upisi sliku sa svim slikama za trening podatke
    #cv2.imwrite("assets/img_log/training-set/training-set.png",samples_image)

