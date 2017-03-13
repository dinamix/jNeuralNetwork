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

import csv

from keras.models import load_model


def main():

	BATCH_SIZE = 128;

	model = load_model('best_model.h5')

	trainX = np.load('tinyX.npy') 
	trainY = np.load('tinyY.npy') 
	testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

	Y = np_utils.to_categorical(trainY, 40);

	Y = Y[:len(testX)]	

	testX = np.swapaxes(testX, 1, 3);
	testX = np.swapaxes(testX, 1, 2);
	testX = testX.astype(np.float32);

	trainX = np.swapaxes(trainX, 1, 3);
	trainX = np.swapaxes(trainX, 1, 2);
	trainX = trainX.astype(np.float32);

	testGenerator = keras.preprocessing.image.ImageDataGenerator(
		featurewise_center=True,
		samplewise_center=False,
		featurewise_std_normalization=True,
		samplewise_std_normalization=False);

	testGenerator.fit(trainX);
	testFlow = testGenerator.flow(testX, batch_size=BATCH_SIZE, shuffle=False)

	predictions = model.predict_generator(testFlow, 6600)

	formattedPredictions = []

	for idx, prediction in enumerate(predictions):
		formattedPredictions.append([idx, np.argmax(prediction)])

	with open("output.csv", "wt") as f:
		writer = csv.writer(f)
		writer.writerow(["id", "class"])
		writer.writerows(formattedPredictions)


if __name__ == "__main__":
	main()