import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AtrousConvolution2D
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam
from keras.constraints import maxnorm

import csv

from keras.models import load_model


def cnn(X, Y):

	BATCH_SIZE = 128;
	CROSS_VAL = 0.10;
	EPOCH_NUM = 100;
	SOFTMAX_SIZE = 40;

	Y = np_utils.to_categorical(Y, 40);

	X = np.swapaxes(X, 1, 3);
	X = np.swapaxes(X, 1, 2);

	X = X.astype(np.float32);

	indices = int(Y.shape[0] * CROSS_VAL);

	XValid = X[:indices]
	YValid = Y[:indices]

	XTrain = X[indices:]
	YTrain = Y[indices:]

	model = Sequential()

	model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.2))

	model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.2))

	model.add(Dense(SOFTMAX_SIZE))
	model.add(Activation('softmax'))

	adam = Adam(lr=0.0002)
	model.compile(
	    loss='categorical_crossentropy',
	    optimizer=adam,
	    metrics=['accuracy'])

	trainGenerator = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
	    samplewise_center=False,
	    featurewise_std_normalization=True,
	    samplewise_std_normalization=False,
	    zca_whitening=False,
	    rotation_range=20,
	    width_shift_range=0.20,
	    height_shift_range=0.20,
	    shear_range=0.20,
	    zoom_range=0.20,
	    channel_shift_range=0.05,
	    fill_mode='nearest',                                                                                                                                                                                                                                                                        
	    cval=0.,
	    horizontal_flip=True,
	    vertical_flip=False,
	    rescale=None,
	    dim_ordering=K.image_dim_ordering())

	trainGenerator.fit(XTrain);
	trainFlow = trainGenerator.flow(XTrain, YTrain, batch_size=BATCH_SIZE);

	validationGenerator = keras.preprocessing.image.ImageDataGenerator(
		featurewise_center=True,
		samplewise_center=False,
		featurewise_std_normalization=True,
		samplewise_std_normalization=False);

	validationGenerator.fit(XTrain);
	validFlow = validationGenerator.flow(XTrain[0:indices], YTrain[0:indices], batch_size=BATCH_SIZE);

	model.fit_generator(trainFlow,
		samples_per_epoch=XTrain.shape[0] - indices,
		nb_epoch=EPOCH_NUM,
		callbacks=[ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)],
		validation_data=validFlow,
		nb_val_samples=indices)

def initProcessing():	

	trainX = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
	trainY = np.load('tinyY.npy') 
	testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

	randomization = np.random.permutation(trainY.shape[0])

	trainX[:] = trainX[randomization]
	trainY[:] = trainY[randomization]

	model = cnn(trainX, trainY)


if __name__ == "__main__":
	initProcessing()