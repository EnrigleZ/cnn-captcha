import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import time


CAPTCHA_IMAGE_PATH = "./image/"
CAPTCHA_IMAGE_WIDTH = 160
CAPTCHA_IMAGE_HEIGHT = 60

CAPTCHA_SET_LEN = 10
CAPTCHA_LEN = 4

MODEL_SAVE_PATH = "./model/"


TRAINING_SET_PERCENTAGE = 0.6
VALIDATING_SET = []
TRAINING_SET = []


def get_image_filename(image_path=CAPTCHA_IMAGE_PATH):
    list_filename = []
    for filename in os.listdir(image_path):
        list_filename.append(filename.split('/')[-1])
    return list_filename


def name2label(filename):
    filename = filename[:CAPTCHA_LEN]

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


def divide_dataset():
    list_filename = get_image_filename(CAPTCHA_IMAGE_PATH)
    random.seed(time.time())
    random.shuffle(list_filename)
    training_set_size = int(TRAINING_SET_PERCENTAGE * len(list_filename))

    #print(training_set_size)
    return list_filename[:training_set_size], list_filename[training_set_size:]


def get_next_batch(batch_size=64, type='train', step=0):
    batch_data = np.zeros([batch_size, CAPTCHA_IMAGE_WIDTH*CAPTCHA_IMAGE_HEIGHT])
    batch_label = np.zeros([batch_size, CAPTCHA_LEN*CAPTCHA_SET_LEN])
    list_filename = TRAINING_SET
    if type=='validate':
        list_filename = VALIDATING_SET

    total = len(list_filename)
    start = step*batch_size

    for i in range(batch_size):
        index = (start + i) % batch_size
        batch_data[i, :], batch_label[i, :] = get_data_and_label(list_filename[index])

    return batch_data, batch_label


def train_data_with_cnn():
    # initialize weight
    def weight_variable(shape, name='weight'):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

    # initialize bias
    def bias_variable(shape, name='bias'):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(initial_value=init, name=name)
        return var

    # convolution
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

    # pooling
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    X = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_WIDTH * CAPTCHA_IMAGE_HEIGHT], name='data-input')
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CAPTCHA_SET_LEN], name='label-input')
    x_input = tf.reshape(X, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDTH, 1], name='X-input')

    keep_prob = tf.placeholder(tf.float32, name='keep-prob')

    # first layer
    W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    W_conv3 = weight_variable([5, 5, 64, 64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)


    # fc layer
    W_fc1 = weight_variable([CAPTCHA_IMAGE_WIDTH*CAPTCHA_IMAGE_HEIGHT/64*64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, CAPTCHA_IMAGE_WIDTH*CAPTCHA_IMAGE_HEIGHT/64*64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)

    W_fc2 = weight_variable([1024, CAPTCHA_SET_LEN*CAPTCHA_LEN], 'W_fc2')
    B_fc2 = bias_variable([CAPTCHA_SET_LEN*CAPTCHA_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CAPTCHA_LEN, CAPTCHA_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, CAPTCHA_LEN, CAPTCHA_SET_LEN], name='labels')

    predict_max_index = tf.argmax(predict, axis=2, name='predict_max_index')
    labels_max_index = tf.argmax(labels, axis=2, name='labels_max_index')

    predict_corrent_vector = tf.equal(predict_max_index, labels_max_index)
    accuracy = tf.reduce_mean(tf.cast(predict_corrent_vector, tf.float32))

    saver = tf.train.Saver()
    


def get_captcha_from_label(label):
    label = label.reshape([CAPTCHA_LEN, CAPTCHA_SET_LEN])
    return np.argmax(label, axis=1)


if __name__ == '__main__':
    TRAINING_SET, VALIDATING_SET = divide_dataset()
    batch_data, batch_label = get_next_batch()

    for index, element in enumerate(batch_label):
        print(index, get_captcha_from_label(element))
