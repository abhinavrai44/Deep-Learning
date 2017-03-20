import numpy as np
import os
import glob
import cv2
import pickle
import datetime
import pandas as pd
import time
from shutil import copy2
import warnings
warnings.filterwarnings("ignore")
from numpy.random import permutation
np.random.seed(2016)
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras.models import load_model
import h5py


def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (224, 224))
    return resized

def load_train():
    X_train = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    for j in range(5):
        print('Load folder c{}'.format(j))
        path = os.path.join('train_224', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im_cv2(fl)
            X_train.append(img)
            y_train.append(j)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train



def read_and_normalize_train_data():
    train_data, train_target = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float16')
    mean_pixel = [103.939, 116.779, 123.68]
    print('Substract 0...')
    train_data[:, 0, :, :] -= mean_pixel[0]
    print('Substract 1...')
    train_data[:, 1, :, :] -= mean_pixel[1]
    print('Substract 2...')
    train_data[:, 2, :, :] -= mean_pixel[2]

    train_target = np_utils.to_categorical(train_target, 10)

    # Shuffle
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target


def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
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

    f = h5py.File('weights/vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def create_model():
    batch_size = 32
    nb_epoch = 4

    X_train, Y_train = read_and_normalize_train_data()

    # model = load_model('save_vgg_model.h5')
    model = VGG_16()

    print('Train: ', len(X_train), len(Y_train))

    X_test = X_train[3900:]
    Y_test = Y_train[3900:]

    X_train = X_train[:3900]
    Y_train = Y_train[:3900]

    callbacks = [
        EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    ]
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          shuffle=True, verbose=1, validation_split = 0.1,
          callbacks=callbacks)
    
    model.save('save_vgg_model.h5')
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Score log_loss: ', score)


if __name__ == '__main__':
    if not os.path.isfile("weights/vgg16_weights.h5"):
        print('Please put VGG16 pretrained weights in weights/vgg16_weights.h5')
        exit()
    create_model()
