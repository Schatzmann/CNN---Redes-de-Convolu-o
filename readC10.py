#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Autor:
#Annelyse Schatzmann         GRR20151731


import cv2
import numpy as np
import os
import pandas
from keras.utils import np_utils

#_______________ CIFAR 10_______________#

import platform
from six.moves import cPickle as pickle

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

def load_pickle(f):
  version = platform.python_version_tuple()
  if version[0] == '2':
    return  pickle.load(f)
  elif version[0] == '3':
    return  pickle.load(f, encoding='latin1')
  

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    
    X = X.reshape(10000,32,32,3)
    Y = np.array(Y)

  return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    
  return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
  # Load the raw CIFAR-10 data
  cifar10_dir = './cifar-10-batches-py/'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # Subsample the data
  label = range(num_training, num_training + num_validation)
  label = range(num_training)
  X_train = X_train[label]
  y_train = y_train[label]
  label = range(num_test)
  X_test = X_test[label]
  y_test = y_test[label]

  X_train /= 255
  X_test /= 255
  
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')

  # convert class vectors to binary class matrices
  Y_train = np_utils.to_categorical(y_train, 10)
  Y_test = np_utils.to_categorical(y_test, 10)



  return X_train, Y_train, X_test, Y_test