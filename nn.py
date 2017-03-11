import numpy as np
import csv
import cv2
from sklearn.preprocessing import normalize
from keras.utils import np_utils
from sklearn.utils import shuffle

class Network(object):

	def __init__(self, sizes):

		self.sizes = sizes
		self.nOfLayers = len(sizes)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def sigmoid(z):
	    return 1.0 / (1.0 + np.exp(-z))

	# lecture 14 p.8
	def feedForwad(self, a):

		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)

		return a

	def gradientDescent(self, trainData, epochs, batchSize, learningRate, validationX = None):

		if validationX:
			validationLen = len(validationX)

		trainLen = len(trainData)

		for i in xrange(epochs):

			trainData = shuffle(trainData)

			#create batches of size batchSize
			batches = [trainData[k:k+batchSize] for k in xrange(0, trainLen, batchSize)]

			for batch in batches:
				self.updateBatches(batch, learningRate)

			print "Epoch: " + str(i)

			if (validationX):
				self.evaluate(validationX)








def flattenImages(data):
	print "Flattening images..."
	flattenedImages = np.array([ np.array(img).flatten() for img in data ])
	print "Finished flattening images..."
	return flattenedImages

def grayImages(data):
	print "Graying images..."
	grayImages = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data])
	print "Finished graying images..."
	return grayImages

def normalize(X):
	print "Normalizing images..."
	X =  X / 255.0
	print "Finished normalizing images..."
	return X

def initProcessing():

	trainX = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
	trainY = np.load('tinyY.npy') 
	testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

	trainY = np_utils.to_categorical(trainY, 40);

	# Convert 3x64x64 to 64x64x3
	trainX = np.rollaxis(trainX, 1, 4)
	testX = np.rollaxis(testX, 1, 4)

	# (trainX, trainY) = shuffle(trainX, trainY, random_state=0)

	trainXData = grayImages(trainX)
	testXData = grayImages(testX)

	trainXData = flattenImages(trainXData)
	testXData = flattenImages(testXData)	

	trainXData = normalize(trainXData)
	testXData = normalize(testXData)

	return trainXData, trainY, testXData

def main():

	(trainX, trainY, testX) = initProcessing()

	network = Network([1, 2, 40])



if __name__ == "__main__":
	main()