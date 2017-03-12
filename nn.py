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
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	# lecture 14 p.12
	def sigmoid_prime(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	# Lecture 14 p.12
	def cost_derivative(self, output_activations, y):

		output_activations = output_activations.reshape(-1, 1)
		y = y.reshape(-1, 1)

		return (output_activations - y)

	# lecture 14 p.8
	def feedForward(self, a):

		activation = a
		activations = [a]
		zs = []

		for w in self.weights:
			z = np.dot(w, activation)
			zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)

		return activations, zs

	def gradientDescent(self, trainData, epochs, batchSize, learningRate, validationX = None):

		for i in xrange(epochs):

			self.updateBatches(trainData, learningRate)

			print "Epoch: " + str(i)

			if (validationX):
				self.evaluate(validationX)


	def updateBatches(self, batch, learningRate):
		
		ww = [np.zeros(w.shape) for w in self.weights]
		
		for x, y in batch:
			dw = self.backpropagation(x, y)
			ww = [nw + dnw for nw, dnw in zip(ww, dw)]

		self.weights = [w - (learningRate / len(batch)) * nw for w, nw in zip(self.weights, ww)]

	# Lecture 14 p.16
	def backpropagation(self, x, y):
		ww = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		(activations, zs) = self.feedForward(x)

		# backward pass
		# lecture 14 p.12: cost derivation * sigmoid prime
		delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1]).reshape(-1, 1)
		ww[-1] = np.dot(delta, activations[-2].reshape(1, -1))

		# Lecture 14 p.16
		for l in xrange(2, self.nOfLayers):
			z = zs[-l]
			sp = self.sigmoid_prime(z).reshape(-1 ,1)
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
			ww[-l] = np.dot(delta, activations[-l - 1].reshape(1, -1))

		return ww

	def evaluate(self, test_data):

		test_results = []

		for x, y in test_data:
			(activations, zs0) = self.feedForward(x)
			test_results.append((np.argmax(activations[-1]), np.argmax(y)))

		score = 0

		for (x, y) in test_results:
			if x == y:
				score += 1

		print float(score) / float(len(test_data))


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

	(trainX, trainY) = shuffle(trainX, trainY, random_state=0)

	trainXData = grayImages(trainX)
	testXData = grayImages(testX)

	trainXData = flattenImages(trainXData)
	testXData = flattenImages(testXData)	

	trainXData = normalize(trainXData)
	testXData = normalize(testXData)

	return trainXData, trainY, testXData

def main():

	CROSS_VAL = 0.20;

	(trainX, trainY, testX) = initProcessing()

	indices = int(trainY.shape[0] * CROSS_VAL);

	XValid = trainX[:indices]
	YValid = trainY[:indices]

	XTrain = trainX[indices:]
	YTrain = trainY[indices:]

	network = Network([4096, 100, 100, 40])

	trainData = zip(XTrain, YTrain)
	validData = zip(XValid, YValid)

	#def gradientDescent(self, trainData, epochs, batchSize, learningRate, validationX = None):
	network.gradientDescent(trainData, 100, 128, 0.001, validationX=validData)




if __name__ == "__main__":
	main()