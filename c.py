import numpy as np
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation



def cnn(X, Y):

	X_test = X[len(X)*80:]
	Y_test = Y[len(Y)*80:]

	X = X[:len(X)*80]
	Y = Y[:len(Y)*80]

	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	img_aug = ImageAugmentation()
	img_aug.add_random_blur(sigma_max=3.)
	
	network = input_data(shape=[None, 64, 64, 3], 
				data_preprocessing=img_prep,
				data_augmentation=img_aug)

	network = conv_2d(network, 32, 3, activation='relu')

	network = max_pool_2d(network, 2)

	network = conv_2d(network, 64, 3, activation='relu')

	network = conv_2d(network, 64, 3, activation='relu')
	
	network = max_pool_2d(network, 2)

	network = fully_connected(network, 512, activation='relu')

	network = dropout(network, 0.5)

	network = fully_connected(network, 2, activation='softmax')

	network = regression(network, optimizer='adam',
				loss='categorical_crossentropy',
				learning_rate=0.001)

	model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='img-classifier.tfl.ckpt')

	print "Fitting..."
	model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
				show_metric=True, batch_size=96,
				snapshot_epoch=True,
				run_id='img-classifier')

	print "Finished fitting..."

	model.save("img-classifier.tfl")

	return model

def format(data):

	return [img / 255.0 for img in data]



def initProcessing():	

	trainX = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
	trainY = np.load('tinyY.npy') 
	testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

	# Convert 3x64x64 to 64x64x3
	trainX = np.rollaxis(trainX, 1, 4)
	testX = np.rollaxis(testX, 1, 4)

	trainX = format(trainX)
	testX = format(testX)

	trainX, trainY = shuffle(trainX, trainY)

	model = cnn(trainX, trainY)

	# predictions = model.predict(testX)


if __name__ == "__main__":
	initProcessing()