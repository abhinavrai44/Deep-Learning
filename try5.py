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

DATASET_PATH = os.environ.get('DATASET_PATH', 'c0.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    train_dataset, train_labels = pickle.load(f)


print('Training set', train_dataset.shape, train_labels.shape)
print (train_labels[22])

DATASET_PATH = os.environ.get('DATASET_PATH', 'c1.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)

DATASET_PATH = os.environ.get('DATASET_PATH', 'c2.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)


DATASET_PATH = os.environ.get('DATASET_PATH', 'c3.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)
  

DATASET_PATH = os.environ.get('DATASET_PATH', 'c4.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)
  

DATASET_PATH = os.environ.get('DATASET_PATH', 'c5.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

DATASET_PATH = os.environ.get('DATASET_PATH', 'c6.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)
  

DATASET_PATH = os.environ.get('DATASET_PATH', 'c7.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)
  

DATASET_PATH = os.environ.get('DATASET_PATH', 'c8.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)
  

DATASET_PATH = os.environ.get('DATASET_PATH', 'c9.pkl')

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    dataset, labels = pickle.load(f)

 

train_dataset = np.concatenate((train_dataset,dataset), axis=0)
train_labels = np.concatenate((train_labels,labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)

del dataset
del labels
  

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:,:]
	shuffled_labels = labels[permutation,:]
	return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)

print('Training set', train_dataset.shape, train_labels.shape)

X_train = train_dataset[:50400]
Y_train = train_labels[:50400]
X_test = X_train[50400:53760]
Y_test = Y_train[50400:53760]

X_val = train_dataset[53760:]
Y_val = train_labels[53760:]

del train_dataset
del train_labels

np.random.seed(1337)  # for reproducibility
num_classes = 10

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras import backend as K
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(100, 100, 3)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))

# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

epochs = 25
lrate = 0.002
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=1, batch_size=64)

model.save('save_aug.h5')
# Final evaluation of the model
scores = model.evaluate(X_val, Y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

output = model.predict(X_test, batch_size=32, verbose=0)

count = [0] * 10

for i in range(400):
    pos = np.argmax(output[i])
    count[pos] += 1

print(count)