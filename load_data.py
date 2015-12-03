#!/usr/bin/env python

import os
import sys

from skimage.io import imread, resize
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_bees():
    '''
    helper function to load our data
    '''
    home_dir = "/home/ubuntu/bee_images/train"
    labels = "/home/ubuntu/bee_images"
    train_labels = pd.read_csv(labels + '/' + "train_labels.csv")
    train_labels.set_index('id', inplace = True)

    bee_images = os.listdir(home_dir)
    bee_images = filter(lambda f: f[-3:] == 'jpg', bee_images)
    bee_images = filter(lambda f: f != '1974.jpg', bee_images)

    bees = []
    for i in bee_images:
        im = imread(home_dir + "/" + i, as_grey = False)
        im = resize(im, (48, 48))
        bees.append(im)

    # divide bees by 255 to give it a 0 - 1 scale
    # (255 is the current max val and zero is the min)
    bees = np.array(bees)/255.0

    Y = train_labels.ix[[int(x.split('.')[0]) for x in bee_images]].values

    onehot = OneHotEncoder(sparse = False, n_values = 2)

    Y = onehot.fit_transform(Y)

    return gen_data(bees, Y)


def gen_data(images, labels):
    '''
    rotate image in all four directions and flip along vertical axis
    to create additional training data
    '''
    new_data = []
    for im in images:
        for angle in range(0, 360, 90):
            new_data.append(rotate(im, angle))
            new_data.append(np.fliplr(rotate(im, angle)))

    new_labels = []
    for l in labels:
        for i in xrange(8):
            new_labels.append(l)


    return np.array(new_data), np.array(new_labels)


# def balance(X, Y):
#     '''
#     takes an unbalanced dataset and undersamples the over represented class
#     '''
#
