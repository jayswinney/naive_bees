#!/usr/bin/env python

import os
import sys

from skimage.io import imread
from skimage.transform import resize, rotate
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_bees():
    '''
    helper function to load our data
    '''
    train_fp = "/home/ubuntu/bee_images/train"
    labels = "/home/ubuntu/bee_images"
    train_labels = pd.read_csv(labels + '/' + "train_labels.csv")
    train_labels.set_index('id', inplace = True)

    bee_images = os.listdir(train_fp)
    bee_images = filter(lambda f: f[-3:] == 'jpg', bee_images)
    bee_images = filter(lambda f: f != '1974.jpg', bee_images)

    bees = []
    for i in bee_images:
        im = imread(train_fp + "/" + i, as_grey = False)
        im = resize(im, (48, 48))
        bees.append(im)

    # divide bees by 255 to give it a 0 - 1 scale
    # (255 is the current max val and zero is the min)
    bees = np.array(bees)/255.0

    Y = train_labels.ix[[int(x.split('.')[0]) for x in bee_images]].values

    onehot = OneHotEncoder(sparse = False, n_values = 2)

    Y = onehot.fit_transform(Y)
    bees, Y = gen_data(bees, Y)
    return balance(bees, Y)


def gen_data(images, labels):
    '''
    rotate image in all four directions and flip along vertical axis
    to create additional training data
    '''
    new_data = []
    for im in images:
        for angle in range(0, 360, 90):
            # rotate each image up, down, left and right
            # and reflect each direction
            new_data.append(rotate(im, angle))
            new_data.append(np.fliplr(rotate(im, angle)))

    # add the new labels to match up with the new images
    new_labels = []
    for l in labels:
        for i in xrange(8):
            new_labels.append(l)


    return np.array(new_data), np.array(new_labels)


def balance(x, y):
    '''
    takes an unbalanced dataset and undersamples the over represented class
    by selecting the number of images from the smaller class from both classes
    '''

    ones = np.array(range(x.shape[0]))[y[:,0] == 1]
    zeros = np.array(range(x.shape[0]))[y[:,0] == 0]

    ones_count = ones.shape[0]
    zeros_count = zeros.shape[0]

    if ones_count > zeros_count:
        idx = np.append(ones, np.random.choice(zeros, ones_count))
    else:
        idx = np.append(zeros, np.random.choice(ones, zeros_count))

    np.random.shuffle(idx)

    return x[idx], y[idx]
