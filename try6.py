from __future__ import print_function
from __future__ import division
import numpy as np
import os
import glob
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

NUM_CLASSES = 10

import matplotlib.pyplot as plt
import numpy as np
import sys
import tarfile
import random
import hashlib

from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from six.moves import range

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

DATASET_PATH = os.environ.get('DATASET_PATH', 'c0_224.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    train_dataset, train_labels = pickle.load(f)


print('Training set', train_dataset.shape, train_labels.shape)
print (train_labels[22])

DATASET_PATH = os.environ.get('DATASET_PATH', 'c1_224.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)

DATASET_PATH = os.environ.get('DATASET_PATH', 'c2_224.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)


DATASET_PATH = os.environ.get('DATASET_PATH', 'c3_224.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)
  

# DATASET_PATH = os.environ.get('DATASET_PATH', 'c4_224.pkl')

# print('Loading dataset {}...'.format(DATASET_PATH))
# with open(DATASET_PATH, 'rb') as f:
#     dataset, labels = pickle.load(f)

 

# train_dataset = np.concatenate((train_dataset,dataset), axis=0)
# train_labels = np.concatenate((train_labels,labels), axis=0)

# print('Training set', train_dataset.shape, train_labels.shape)
  

# DATASET_PATH = os.environ.get('DATASET_PATH', 'c5_224.pkl')

# print('Loading dataset {}...'.format(DATASET_PATH))
# with open(DATASET_PATH, 'rb') as f:
#     dataset, labels = pickle.load(f)

 

# train_dataset = np.concatenate((train_dataset,dataset), axis=0)
# train_labels = np.concatenate((train_labels,labels), axis=0)

# DATASET_PATH = os.environ.get('DATASET_PATH', 'c6_224.pkl')

# print('Loading dataset {}...'.format(DATASET_PATH))
# with open(DATASET_PATH, 'rb') as f:
#     dataset, labels = pickle.load(f)

 

# train_dataset = np.concatenate((train_dataset,dataset), axis=0)
# train_labels = np.concatenate((train_labels,labels), axis=0)

# print('Training set', train_dataset.shape, train_labels.shape)
  

# DATASET_PATH = os.environ.get('DATASET_PATH', 'c7_224.pkl')

# print('Loading dataset {}...'.format(DATASET_PATH))
# with open(DATASET_PATH, 'rb') as f:
#     dataset, labels = pickle.load(f)

 

# train_dataset = np.concatenate((train_dataset,dataset), axis=0)
# train_labels = np.concatenate((train_labels,labels), axis=0)

# print('Training set', train_dataset.shape, train_labels.shape)
  

# DATASET_PATH = os.environ.get('DATASET_PATH', 'c8_224.pkl')

# print('Loading dataset {}...'.format(DATASET_PATH))
# with open(DATASET_PATH, 'rb') as f:
#     dataset, labels = pickle.load(f)

 

# train_dataset = np.concatenate((train_dataset,dataset), axis=0)
# train_labels = np.concatenate((train_labels,labels), axis=0)

# print('Training set', train_dataset.shape, train_labels.shape)
  

# DATASET_PATH = os.environ.get('DATASET_PATH', 'c9_224.pkl')

# print('Loading dataset {}...'.format(DATASET_PATH))
# with open(DATASET_PATH, 'rb') as f:
#     dataset, labels = pickle.load(f)

 

# train_dataset = np.concatenate((train_dataset,dataset), axis=0)
# train_labels = np.concatenate((train_labels,labels), axis=0)

# print('Training set', train_dataset.shape, train_labels.shape)

del dataset
del labels
  

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:,:]
	shuffled_labels = labels[permutation,:]
	return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)

X_train = train_dataset[:7000]
Y_train = train_labels[:7000]
X_test = X_train[7000:]
Y_test = Y_train[7000:]

del train_dataset
del train_labels

print('Training set', X_train.shape, Y_train.shape)

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

mymodel = VGG_16('vgg16_weights.h5')
for l in range(0,5):
    del mymodel.layers[-1]
mymodel.add(Dense(256, activation='relu'))
mymodel.add(Dense(0.5))
mymodel.add(Dense(256, activation='relu'))
mymodel.add(Dense(0.5))
mymodel.add(Dense(10, activation='softmax'))

unfreeze_last = 5
for num in range(0,len(mymodel.layers)):
    layer = mymodel.layers[num]
    if num < len(mymodel.layers)-unfreeze_last:
        layer.trainable = False
    else:
        layer.trainable = True 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
mymodel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

mymodel.fit(X_train, Y_train, batch_size=64, validation_split=0.1, nb_epoch=1, shuffle=True)
mymodel.save('save_vgg.h5')

scores = mymodel.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

output = model.predict(X_test, batch_size=32, verbose=0)

count = [0] * 10

for i in range(400):
    pos = np.argmax(output[i])
    count[pos] += 1

print(count)