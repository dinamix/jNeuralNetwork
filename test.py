import numpy as np
import cv2
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from skimage.filters import gaussian_filter
from sklearn import svm



sift = cv2.SIFT()

# to visualize only
# import scipy.misc
# scipy.misc.imshow(trainX[0].transpose(2,1,0)) # put RGB channels last

def kFoldTest(trainX, trainY):
	k_fold = KFold(n=len(trainX), n_folds=6)

	for train_indices, test_indices in k_fold:

		train_x = []
		train_y = []
		test_x = []
		test_y = []

		for i in train_indices:
			train_x.append(trainX[i])
			train_y.append(trainY[i])

		for i in test_indices:
			test_x.append(trainX[i])
			test_y.append(trainY[i])			

		predictions = logisticRegression(train_x, train_y, test_x)

		score = f1_score(test_y, predictions, average='macro')
		accuracy = accuracy_score(test_y, predictions)
		report=classification_report(test_y, predictions)
		print score
		print accuracy
		print report

def predict(trainX, trainY, testX):

	print "Started predictions..."

	predictions = logisticRegression(trainX, trainY, testX)

	formattedPredictions = []

	for idx, prediction in enumerate(predictions):
		formattedPredictions.append([idx, prediction])

	with open("output.csv", "wt") as f:
		writer = csv.writer(f)
		writer.writerow(["id", "class"])
		writer.writerows(formattedPredictions)

	print "Finished predictions..."

def logisticRegression(trainX, trainY, testX):

	#logreg = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

	logreg = svm.SVC()

	logreg.fit(trainX, trainY)

	return logreg.predict(testX)


def computeFeatures(images):

	print "Computing features..."

	kds = []
	kps = []    

	for img in images:

		# Grayscaling
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# blurred
		# gray = gaussian_filter(gray, sigma=20)

		# extracting features, whatever that means
		dense = cv2.FeatureDetector_create("Dense")
		kp = dense.detect(gray)

		kp, kd = sift.compute(gray, kp)
		kds.append(kd)
		kps.append(kp)

	print "Computing features..."
        
	return kps, kds


def initProcessing():

	trainX = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
	trainY = np.load('tinyY.npy') 
	testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

	# Convert 3x64x64 to 64x64x3
	trainX = np.rollaxis(trainX, 1, 4)
	testX = np.rollaxis(testX, 1, 4)

	(trainX, trainY) = shuffle(trainX, trainY, random_state=0)

	(trainKps, trainKds) = computeFeatures(trainX)
	(testKps, testKds) = computeFeatures(testX)

	trainXData = []
	trainYData = []
	testXData = []

	for i in range(len(trainKds)):
		trainXData.append(trainKds[i][0] + trainKds[i][1] + trainKds[i][2])
		trainYData.append(trainY[i])

	for i in range(len(testKds)):
		testXData.append(testKds[i][0] + testKds[i][1] + testKds[i][2])

	

	kFoldTest(trainXData, trainYData)

	#predict(trainXData, trainYData, testXData)



if __name__ == "__main__":
	initProcessing()