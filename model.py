import tensorflow as tf
import numpy as np
from PIL import Image
import os


CAPTCHA_IMAGE_PATH = "./image/"
CAPTCHA_IMAGE_WIDTH = 160
CAPTCHA_IMAGE_HEIGHT = 60

CAPTCHA_SET_LEN = 10
CAPTCHA_LEN = 4

MODEL_SAVE_PATH = "./model/"


def get_image_filename(image_path=CAPTCHA_IMAGE_PATH):
    list_filename = []
    for filename in os.listdir(image_path):
        list_filename.append(filename.split('/')[-1])


def name2label(filename):
    assert(len(filename) == CAPTCHA_LEN)

    label = np.zeros(CAPTCHA_LEN*CAPTCHA_SET_LEN)   # 4 * 10
    for index, element in enumerate(filename, 0):
        idx = index*CAPTCHA_SET_LEN + ord(element) - ord('0')
        label[idx] = 1
    return label


def get_data_and_label(filename, dirpath=CAPTCHA_IMAGE_PATH):
    filepath = os.path.join(dirpath, filename)

    img = Image.open(filepath)  # open an image
    img = img.convert("L")      # convert image to grey scale
    image_data = np.array(img).flatten() / 255
    image_label = name2label(filename)

    return image_data, image_label


