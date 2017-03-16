import numpy as np
import csv
import cv2
from sklearn.preprocessing import normalize
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Network(object):

	def __init__(self, sizes):

		self.sizes = sizes
		self.nOfLayers = len(sizes)
		self.weights = [np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])]

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	# lecture 14 p.12
	def sigmoidPrime(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	# Lecture 14 p.12
	def costDerivative(self, output_activations, y):

		output_activations = output_activations.reshape(-1, 1)
		y = y.reshape(-1, 1)

		return (y - output_activations)

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

	def gradientDescent(self, trainData, epochs, learningRate, validationX):

		epochList = []
		accuracyList = []

		for i in xrange(epochs):

			self.updateWeights(trainData, learningRate)

			print "Epoch: " + str(i)

			(predictions, accuracy) = self.predict(validationX)

			epochList.append(i)
			accuracyList.append(accuracy)

		return (epochList, accuracyList)

	# lecture 14 p.16
	def updateWeights(self, data, learningRate):
		
		ww = [np.zeros(w.shape) for w in self.weights]
		
		# Pick a training example
		for x, y in data:

			dw = [np.zeros(w.shape) for w in self.weights]
			# feedforward: Feed example through network to compute 
			(activations, zs) = self.feedForward(x)

			# backprog
			# lecture 14 p.12: cost derivation * sigmoid prime
			# For the output unit, compute the correction
			delta = self.costDerivative(activations[-1], y) * self.sigmoidPrime(zs[-1]).reshape(-1, 1)
			dw[-1] = np.dot(delta, activations[-2].reshape(1, -1))

			# Lecture 14 p.16
			# For each hidden unit h, compute its share of the correction
			for l in xrange(2, self.nOfLayers):
				delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.sigmoidPrime(zs[-l]).reshape(-1 ,1)
				dw[-l] = np.dot(delta, activations[-l - 1].reshape(1, -1))


			ww = [nw + ddw for nw, ddw in zip(ww, dw)]

		self.weights = [w + learningRate * nw for w, nw in zip(self.weights, ww)]


	def predict(self, test_data):

		test_results = []
		test = []

		for x, y in test_data:
			(activations, zs0) = self.feedForward(x)
			test.append(np.argmax(activations[-1]))
			test_results.append((np.argmax(activations[-1]), np.argmax(y)))

		score = 0

		for (x, y) in test_results:
			if x == y:
				score += 1

		print float(score) / float(len(test_data))

		return (test, float(score) / float(len(test_data)))


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

	randomization = np.random.permutation(trainY.shape[0])

	trainX[:] = trainX[randomization]
	trainY[:] = trainY[randomization]

	# trainXData = grayImages(trainX)
	# testXData = grayImages(testX)

	trainXData = flattenImages(trainX)
	testXData = flattenImages(testX)	

	trainXData = normalize(trainXData)
	testXData = normalize(testXData)

	return trainXData, trainY, testXData

def kFoldTest(trainX, trainY):

	print "Starting kFold..."

	k_fold = KFold(n=len(trainX), n_folds=6)

	nOfComponents = 10

	print "Fitting..."
	pca = PCA(n_components=nOfComponents)
	pca.fit(trainX[:2500])
	trainX = pca.transform(trainX)
	print "Done fitting..."

	layers = [		
		# [nOfComponents, 10, 40],
		# [nOfComponents, 15, 40],
		# [nOfComponents, 20, 40],
		[nOfComponents, 25, 40]
		# [nOfComponents, 30, 40],
		# [nOfComponents, 35, 40]
	]

	for layer in layers:

		print "Layer..." 


		# for train_indices, test_indices in k_fold:

		train_x = []
		train_y = []
		test_x = []
		test_y = []

		train_x = trainX[:int(len(trainX) * 0.80)]
		train_y = trainY[:int(len(trainY) * 0.80)]

		test_x = trainX[int(len(trainX) * 0.80):]
		test_y = trainX[int(len(trainY) * 0.80):]

		# for i in train_indices:
		# 	train_x.append(trainX[i])
		# 	train_y.append(trainY[i])

		# for i in test_indices:
		# 	test_x.append(trainX[i])
		# 	test_y.append(trainY[i])		

		trainData = zip(train_x, train_y)
		validData = zip(test_x, test_y)

		test_Y = []

		for y in test_y:
			test_Y.append(np.argmax(y))

		network = Network(layer)
		(epochList, accuracyList) = network.gradientDescent(trainData, 50, 0.0005, validationX=validData)
		(predictions, accuracy) = network.predict(validData)

		plt.plot(epochList, accuracyList, markersize=5, label='$Nb of nodes = {a}$'.format(a=layer[1]))			

		score = f1_score(test_Y, predictions, average='macro')
		accuracy = accuracy_score(test_Y, predictions)
		report=classification_report(test_Y, predictions)
		print score
		print accuracy
		print report

		# print "Finished kFold..."

	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs epoch")
	plt.show()

def main():

	CROSS_VAL = 0.30;

	(trainX, trainY, testX) = initProcessing()

	randomization = np.random.permutation(trainY.shape[0])

	trainX[:] = trainX[randomization]
	trainY[:] = trainY[randomization]

	# indices = int(trainY.shape[0] * CROSS_VAL);

	# XValid = trainX[:indices]
	# YValid = trainY[:indices]

	# XTrain = trainX[indices:]
	# YTrain = trainY[indices:]

	# network = Network([4096 * 3, 50, 50, 40])

	# trainData = zip(XTrain, YTrain)
	# validData = zip(XValid, YValid)

	#def gradientDescent(self, trainData, epochs, learningRate, validationX = None):
	# network.gradientDescent(trainData, 100, 0.0001, validationX=validData)

	kFoldTest(trainX[:int(len(trainX) * 0.80)], trainY[:int(len(trainY) * 0.80)])





if __name__ == "__main__":
	main()