#!/usr/bin/env python

import os
import sys

import tensorflow as tf
from skimage.io import imread, imshow
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_bees():
    '''
    helper function to load our data
    '''
    out = "/home/ubuntu/CNN_tensor_flow_output.txt"
    home_dir = "/home/ubuntu/bee_images/train"
    labels = "/home/ubuntu/bee_images"
    train_labels = pd.read_csv(labels + '/' + "train_labels.csv")
    train_labels.set_index('id', inplace = True)

    bee_images = os.listdir(home_dir)
    bee_images = filter(lambda f: f[-3:] == 'jpg', bee_images)
    bee_images = filter(lambda f: f != '1974.jpg', bee_images)

    bees = []
    for i in bee_images:
        bees.append(imread(home_dir + "/" + i, as_grey = False))

    # divide bees by 255 to give it a 0 - 1 scale
    # (255 is the current max val and zero is the min)
    bees = np.array(bees)/255.0

    Y = train_labels.ix[[int(x.split('.')[0]) for x in bee_images]].values

    onehot = OneHotEncoder(sparse = False, n_values = 2)

    Y = onehot.fit_transform(Y)

    return bees, Y
