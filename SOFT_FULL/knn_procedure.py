import numpy as np
import cv2

img_size = 20
num_classes = 10

knn_model = cv2.ml.KNearest_create()

def split_digits_image(digits_image, split_size):
    h,w = digits_image.shape[:2]
    print("Visina i sirina slike: " + str(h) + ", " + str(w))
    x,y = split_size
    print("Visina i sirina delova: " + str(y) + ", " + str(x))
    cells = []
    # np.vsplit(matrica, visina)
    # np.hsplit(matrica, sirina)
    vsplitres = np.vsplit(digits_image, h // y)
    for row in vsplitres:
        hsplitres = np.hsplit(row, w//x)
        for col in hsplitres:
            cells.append(col)
    print("Ukupno ucitanih slika: " + str(len(cells)))
    return cells

def read_digits_image(filepath):
    print("Ucitavanje slike: ["+filepath+"] ...")
    digits_image = cv2.imread(filepath, 0)
    digits = split_digits_image(digits_image, (img_size, img_size) )
    labels = np.repeat(np.arange(num_classes), len(digits) / num_classes)
    return digits, labels

def deskew(img):
    #prepravljanje slike ako je nakosena
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1,skew, -0.5*img_size*skew], [0,1,0]])
    img = cv2.warpAffine(img, M, (img_size, img_size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess(digit_images):
    samples = []
    for image in digit_images:
        thresholded_digit = np.zeros(image.shape, image.dtype)
        cv2.threshold(image, 255,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY, thresholded_digit)
        samples.append(thresholded_digit)
    return samples

def preprocess_and_slice(thresholded_digit_images):
    sliced_images = []
    for digit_image in thresholded_digit_images:
        h, w = digit_image.shape[:2]
        top_idx = -1
        bottom_idx = -1
        left_idx = -1
        right_idx = -1

        #odsecanje gornjeg dela
        for row in range(h):
            for col in range(w):
                if digit_image[row][col] != 0:
                    top_idx = row
                    break
            if top_idx != -1:
                break

        #odsecanje donjeg dela
        for row in reversed(range(h)):
            for col in range(w):
                if digit_image[row][col] != 0:
                    bottom_idx = row
                    break
            if bottom_idx != -1:
                break

        #odsecanje levog dela
        for col in range(w):
            for row in range(h):
                if digit_image[row][col] != 0:
                    left_idx = col
                    break
            if left_idx != -1:
                break

        #odsecanje desnog dela
        for col in reversed(range(w)):
            for row in range(h):
                if digit_image[row][col] != 0:
                    right_idx = col
                    break
            if right_idx != -1:
                break

        sliced_digit_image = digit_image[top_idx:bottom_idx, left_idx:right_idx]
        sliced_digit_image = cv2.resize(sliced_digit_image, (img_size, img_size))
        ret, sliced_digit_image = cv2.threshold(sliced_digit_image, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
        sliced_images.append(sliced_digit_image)

    return sliced_images

def preprocess_slice_erode(thresholded_digit_images):
    sliced_images = []
    for digit_image in thresholded_digit_images:
        h, w = digit_image.shape[:2]
        top_idx = -1
        bottom_idx = -1
        left_idx = -1
        right_idx = -1

        #odsecanje gornjeg dela
        for row in range(h):
            for col in range(w):
                if digit_image[row][col] != 0:
                    top_idx = row
                    break
            if top_idx != -1:
                break

        #odsecanje donjeg dela
        for row in reversed(range(h)):
            for col in range(w):
                if digit_image[row][col] != 0:
                    bottom_idx = row
                    break
            if bottom_idx != -1:
                break

        #odsecanje levog dela
        for col in range(w):
            for row in range(h):
                if digit_image[row][col] != 0:
                    left_idx = col
                    break
            if left_idx != -1:
                break

        #odsecanje desnog dela
        for col in reversed(range(w)):
            for row in range(h):
                if digit_image[row][col] != 0:
                    right_idx = col
                    break
            if right_idx != -1:
                break

        sliced_digit_image = digit_image[top_idx:bottom_idx, left_idx:right_idx]
        sliced_digit_image = cv2.resize(sliced_digit_image, (img_size, img_size))
        ret, sliced_digit_image = cv2.threshold(sliced_digit_image, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
        eroded_slice = cv2.morphologyEx(sliced_digit_image, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
        sliced_images.append(eroded_slice)

    return sliced_images

def flatten_image(image):
    flat_image = image.reshape(1, img_size * img_size)
    return flat_image

def flatten_images(digit_images_thresholded):
    sample_image = np.zeros([len(digit_images_thresholded),img_size*img_size],dtype=np.float32)
    for img_idx in range(len(digit_images_thresholded)):
        flat_image = digit_images_thresholded[img_idx].reshape(1, img_size* img_size)
        sample_image[img_idx] = flat_image
    return sample_image

def get_digit_from_knn(image):
    ret, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    deskewed_image = deskew(thresholded_image)
    sliced_image = preprocess_slice_erode([deskewed_image])[0]
    flat_image = flatten_image(sliced_image)
    flat_image = np.float32(flat_image)

    ret,res,neig,dist = knn_model.findNearest(flat_image, 5)
    return ret

def recognize_contour(contour_to_add, thresholded_frame):
    x, y, w, h = cv2.boundingRect(contour_to_add)
    cropped_image = thresholded_frame[y:y + h, x:x + w]
    resized_cropped_image = cv2.resize(cropped_image, (img_size, img_size))
    digit = get_digit_from_knn(resized_cropped_image)
    return digit

def init_knn(filepath):
    training_set = 2
    try:
        print('Pokusaj ucitavanja gotovog skupa za trening ...')
        if training_set == 1:
            samples_image = cv2.imread("training/training-set.png", 0)
        elif training_set == 2:
            samples_image = cv2.imread("training/training-set-no-deskew.png", 0)
        elif training_set == 3:
            samples_image = cv2.imread("training/training-set-no-deskew-erode.png", 0)
        elif training_set == 4:
            samples_image = cv2.imread("training/training-set-everything.png", 0)
        cv2.imshow('training-set', samples_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        samples_image = np.float32(samples_image)
        labels = np.repeat(np.arange(num_classes), 5000 / num_classes)
        print('Uspesno ucitan trening skup! ...')
    except:
        print('Ne postoji trening skup! ...')
        print('Kreiranje novog ...')
        #ucitavanje
        [digits, labels] = read_digits_image(filepath)
        if training_set == 1:
            digs_straight = list(map(deskew,digits))
            digits_thresholded = preprocess(digs_straight)
            samples_image = flatten_images(digits_thresholded)
            cv2.imwrite("training/training-set.png", samples_image)
        elif training_set == 2:
            digits_thresholded = preprocess(digits)
            sliced_digit_images = preprocess_and_slice(digits_thresholded)
            samples_image = flatten_images(sliced_digit_images)
            cv2.imwrite('training/training-set-no-deskew.png', samples_image)
        elif training_set == 3:
            digits_thresholded = preprocess(digits)
            sliced_digit_images = preprocess_slice_erode(digits_thresholded)
            samples_image = flatten_images(sliced_digit_images)
            cv2.imwrite('training/training-set-no-deskew-erode.png', samples_image)
        elif training_set == 4:
            digits_straight = list(map(deskew, digits))
            digits_thresholded = preprocess(digits_straight)
            sliced_digit_images = preprocess_slice_erode(digits_thresholded)
            samples_image = flatten_images(sliced_digit_images)
            cv2.imwrite('training/training-set-everything.png', samples_image)

    knn_model.train(samples_image, cv2.ml.ROW_SAMPLE, labels)
