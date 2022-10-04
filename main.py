import PIL
import numpy as np
from PIL import Image
from pytesseract import pytesseract
import os
import cv2
import shutil

# Define path to tessaract.exe
path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define path to image
path_to_images = r"C:\Users\Pedro Muniz\Desktop\João PowerPoint\imagens\\"
# print(os.listdir(path_to_images))

# Point tessaract_cmd to tessaract.exe
pytesseract.tesseract_cmd = path_to_tesseract

CWD = os.path.dirname(os.path.realpath(__file__))

left_percentage = 60 / 473
right_percentage = 230 / 473
top_percentage = 17 / 842
bottom_percentage = 53 / 842


def main():
    # Iterate over each file_name in the folder
    names = []
    images = []
    for file_name in os.listdir(path_to_images):
        # Open image with PIL
        if os.path.isdir(path_to_images + file_name):
            continue

        thresh = preprocess(path_to_images + file_name, False)

        # Extract text from image
        text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
        text = text.split("\n")[0]
        text = "".join(x for x in text if x.isalnum())

        names.append(text.split("\n")[0])
        images.append(path_to_images + file_name)

        # print(text)

    name_groups, image_groups = sort_names(names, images)
    save_images(name_groups, image_groups)


def preprocess(path_to_image, show=False):
    img = Image.open(path_to_image)

    left = img.width * left_percentage
    top = img.height * top_percentage
    right = img.width * right_percentage
    bottom = img.height * bottom_percentage

    # crop the image
    new_img = img.crop((left, top, right, bottom))

    pil_image = new_img.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    gamma_corrected = adjust_gamma(gray, 9, True)
    # blur = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)
    # blur = cv2.medianBlur(gamma_corrected, 3)
    blur = cv2.bilateralFilter(gamma_corrected, 5, 75, 75)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    thresh = 255 - thresh
    blur = 255 - blur

    if show:
        cv2.imshow('normal', open_cv_image)
        cv2.imshow('gamma', gamma_corrected)
        cv2.imshow("blur", blur)
        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return blur


def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))
    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
    a = 0
    b = 0
    c = 0
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


# TODO put code to sort names that return "" and take them out of the lists
def sort_names(names, images):
    name_groups = [[name] for name in names]
    image_groups = [[image] for image in images]
    taken = [False for _ in names]
    for i in range(len(names)):
        taken[i] = True
        for j in range(i, len(names)):
            # print("i, j:", i, j)
            if i == j or taken[j]:
                continue
            name1 = names[i]
            name2 = names[j]
            dist = levenshtein_distance(name1, name2)
            percentage = (len(name1) - len(name2)) / max(len(names[i]), len(names[j])) * 100
            if dist <= 5:
                taken[j] = True
                for n in name_groups[j]:
                    name_groups[i].append(n)
                for img in image_groups[j]:
                    image_groups[i].append(img)
                name_groups[j] = []
                image_groups[j] = []
            # print("name1:", name1)
            # print("name2:", name2)
            # print("distance:", dist)
            # print("percentage:", percentage)
    return name_groups, image_groups


def adjust_gamma(image, gamma, binary=False):
    copy_img = image.copy()
    for y in range(copy_img.shape[0]):
        for x in range(copy_img.shape[1]):
            if binary:
                copy_img[y, x] = np.clip(((copy_img[y, x] / 255.0) ** gamma) * 255, 0, 255)
            else:
                for c in range(copy_img.shape[2]):
                    copy_img[y, x, c] = np.clip(((copy_img[y, x, c] / 255.0) ** gamma) * 255, 0, 255)
    return copy_img


def save_images(names, images):
    for i in range(len(names)):
        # create folder
        folder = ""
        if names[i]:
            folder = CWD + "\\" + names[i][0]
            print("FOLDER", folder)
            os.mkdir(CWD + "\\" + names[i][0])
        for j in range(len(names[i])):
            print("IMAGE PATH", images[i][j])
            src = images[i][j]
            dest = rf"{folder}\\{os.path.basename(images[i][j])}"
            print("SRC", src)
            print("DEST", dest)
            shutil.copyfile(src, dest)


if __name__ == '__main__':
    main()
