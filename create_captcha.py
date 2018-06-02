import os
import shutil
import random
import time
from captcha.image import ImageCaptcha

CAPTCHA_IMAGE_PATH = "./image/"
CAPTCHA_TEST_PATH = "./test/"
CAPTCHA_SET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CAPTCHA_LEN = 4


def generate_captcha_image(captcha_image_path=CAPTCHA_IMAGE_PATH,
                           captcha_set=CAPTCHA_SET,
                           captcha_len=CAPTCHA_LEN):
    captcha_set_len = len(captcha_set)
    total = 1
    for i in range(captcha_len):
        total *= captcha_set_len

    cnt_generate = 0

    for index_0 in range(captcha_set_len):
        for index_1 in range(captcha_set_len):
            for index_2 in range(captcha_set_len):
                for index_3 in range(captcha_set_len):
                    captcha_text = captcha_set[index_0] + captcha_set[index_1] + captcha_set[index_2] + captcha_set[index_3]
                    cnt_generate += 1
                    image = ImageCaptcha()
                    image.write(captcha_text, captcha_image_path + captcha_text + '.jpg', format='jpeg')
                print(index_0, index_1, index_2)

# use those images to get accuracy
TEST_IMAGE_NUM = 50

def create_test_set(test_image_num=TEST_IMAGE_NUM):
    list_filename = []
    for filename in os.listdir(CAPTCHA_IMAGE_PATH):
        captcha_name = filename.split('/')[-1]
        list_filename.append(captcha_name)

    random.seed(time.time())
    random.shuffle(list_filename)
    for i in range(test_image_num):
        shutil.move(CAPTCHA_IMAGE_PATH+list_filename[i], CAPTCHA_TEST_PATH+list_filename[i])

if __name__ == '__main__':
    print("Invoke create_test_set() or generate_captcha_image()")