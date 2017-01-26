import pickle
import tflearn
from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import datetime
from config import modelsPath, datasetPath, datasetImageSize, datasetMaxSerieLength
from datasetTools import padLSTM
import numpy as np
import tensorflow as tf
import sys 
import h5py

def createModel(nbClasses, imageSize, maxlength):
	print "======== Creating model... ========"
	print "Creating left and right CNNs..."
	net0 = createCNN(imageSize,maxlength)
	net1 = createCNN(imageSize,maxlength)
	
	print "Creating LSTM..."
	net = createLSTM(net0, net1)

	print "Creating Softmax..."
	net = fully_connected(net, nbClasses, activation='softmax')
	net = regression(net, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.0005)

	print "Putting the model together..."
	model = tflearn.DNN(net, tensorboard_verbose=0)
	print "======== Model created ============"

	return model

#CNN for Right or Left eye image processing
def createCNN(imageSize, maxlength):
	net = input_data(shape=[None, maxlength, imageSize, imageSize, 1])
	net = tflearn.time_distributed(net, conv_2d, [16, 3, 1, 'same', 'linear', True, 'Xavier'])
	net = tflearn.time_distributed(net, fully_connected, [128, 'elu', True, "Xavier"])
	net = tflearn.time_distributed(net, dropout, [0.7])
	return net

#LSTM from Right and Left eye features
def createLSTM(cnn0, cnn1):
	#Merge tensors to concat features
	net = tf.concat(2,[cnn0, cnn1])
	net = tflearn.time_distributed(net, fully_connected, [128, 'elu', True, "Xavier"])	
	net = tflearn.lstm(net, 128, dynamic=False)
	net = dropout(net, 0.7)
	return net

#Trains the model
def trainModel():
	runId = "ZZU - Slide HDF5" + str(datetime.datetime.now().time())

	#Load pickle dataset
	# X0 = pickle.load(open(datasetPath+"X0_hd.p", "rb" ))
	# X1 = pickle.load(open(datasetPath+"X1_hd.p", "rb" ))
	# y = pickle.load(open(datasetPath+"y_hd.p", "rb" ))
	# print len(y), "series, first has shape", X0[0].shape

	# Load hdf5 dataset
	h5f = h5py.File(datasetPath+"data.h5", 'r')
	X0 = h5f['eye_X0']
	X1 = h5f['eye_X1']
	y = h5f['eye_Y']

	lengths = [serie.shape[0] for serie in X0]
	minLength, maxlength = min(lengths), max(lengths)
	print "Lengths range: {} to {}".format(minLength,maxlength)

	X0 = padLSTM(X0, maxlen=maxlength, value=0.)
	X1 = padLSTM(X1, maxlen=maxlength, value=0.)
	print len(X0), "padded series of shape", X0[0].shape

	#Create, fit and save model
	model = createModel(nbClasses=3, imageSize=datasetImageSize, maxlength=maxlength)
	model.fit([X0, X1], y, n_epoch=13, batch_size=80, shuffle=True, validation_set=0.15, show_metric=True, run_id=runId)
	model.save(modelsPath+"eyeDNN_HD_SLIDE.tflearn")

#Tests the model on custom dataset
def testModel():
	#TODO do same with h5f
	#Load test dataset
	print "Loading test dataset..."
	X0_test = pickle.load(open(datasetPath+"X0_hd_test.p", "rb" ))
	X1_test = pickle.load(open(datasetPath+"X1_hd_test.p", "rb" ))
	y_test = pickle.load(open(datasetPath+"y_hd_test.p", "rb" ))

	print len(X0_test), "series of shape", X0_test[0].shape

	X0_test = padLSTM(X0_test, maxlen=datasetMaxSerieLength, value=0.)
	X1_test = padLSTM(X1_test, maxlen=datasetMaxSerieLength, value=0.)

	# Load model
	model = createModel(nbClasses=3, imageSize=datasetImageSize, maxlength=maxlength)
	print "Loading model parameters..."
	model.load(modelsPath+"eyeDNN_HD.tflearn")

	# Test
	print "Making predictions..."
	print model.predict([X0_test,X1_test])
	print y_test

#Main
if __name__ == "__main__":
	arg = sys.argv[1] if len(sys.argv) == 2 else None 
	if arg is None:
		print "Need argument"
	elif arg == "train":
		trainModel()
	elif arg == "test":
		testModel()
	else:
		print "Wrong argument"


