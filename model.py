import pickle
import tflearn
from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import datetime
from config import models_path, dataset_path, dataset_img_size, dataset_max_serie_length
from dataset_tools import pad_lstm
import numpy as np
import tensorflow as tf
import sys 
import h5py

def create_model(nb_classes, img_size, max_length):
	print "======== Creating model... ========"
	print "Creating left and right CNNs..."
	net0 = create_cnn(img_size,max_length)
	net1 = create_cnn(img_size,max_length)
	
	print "Creating LSTM..."
	net = create_lstm(net0, net1)

	print "Creating Softmax..."
	net = fully_connected(net, nb_classes, activation='softmax')
	net = regression(net, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.0005)

	print "Putting the model together..."
	model = tflearn.DNN(net, tensorboard_verbose=0)
	print "======== Model created ============"

	return model

#CNN for Right or Left eye image processing
def create_cnn(img_size, max_length):
	net = input_data(shape=[None, max_length, img_size, img_size, 1])
	net = tflearn.time_distributed(net, conv_2d, [16, 3, 1, 'same', 'linear', True, 'Xavier'])
	net = tflearn.time_distributed(net, fully_connected, [128, 'elu', True, "Xavier"])
	net = tflearn.time_distributed(net, dropout, [0.7])
	return net

#LSTM from Right and Left eye features
def create_lstm(cnn0, cnn1):
	#Merge tensors to concat features
	net = tf.concat(2,[cnn0, cnn1])
	net = tflearn.time_distributed(net, fully_connected, [128, 'elu', True, "Xavier"])	
	net = tflearn.lstm(net, 128, dynamic=False)
	net = dropout(net, 0.7)
	return net

#Trains the model
def train_model():
	runId = "ZZU - Slide HDF5" + str(datetime.datetime.now().time())

	#Load pickle dataset
	# X0 = pickle.load(open(dataset_path+"X0_hd.p", "rb" ))
	# X1 = pickle.load(open(dataset_path+"X1_hd.p", "rb" ))
	# y = pickle.load(open(dataset_path+"y_hd.p", "rb" ))
	# print len(y), "series, first has shape", X0[0].shape

	# Load hdf5 dataset
	h5f = h5py.File(dataset_path+"data.h5", 'r')
	X0 = h5f['eye_X0']
	X1 = h5f['eye_X1']
	y = h5f['eye_Y']

	lengths = [serie.shape[0] for serie in X0]
	minLength, max_length = min(lengths), max(lengths)
	print "Lengths range: {} to {}".format(minLength,max_length)

	X0 = pad_lstm(X0, maxlen=max_length, value=0.)
	X1 = pad_lstm(X1, maxlen=max_length, value=0.)
	print len(X0), "padded series of shape", X0[0].shape

	#Create, fit and save model
	model = create_model(nb_classes=3, img_size=dataset_img_size, max_length=max_length)
	model.fit([X0, X1], y, n_epoch=13, batch_size=80, shuffle=True, validation_set=0.15, show_metric=True, run_id=runId)
	model.save(models_path+"eyeDNN_HD_SLIDE.tflearn")

#Tests the model on custom dataset
def test_model():
	#TODO do same with h5f
	#Load test dataset
	print "Loading test dataset..."
	X0_test = pickle.load(open(dataset_path+"X0_hd_test.p", "rb" ))
	X1_test = pickle.load(open(dataset_path+"X1_hd_test.p", "rb" ))
	y_test = pickle.load(open(dataset_path+"y_hd_test.p", "rb" ))

	print len(X0_test), "series of shape", X0_test[0].shape

	X0_test = pad_lstm(X0_test, maxlen=dataset_max_serie_length, value=0.)
	X1_test = pad_lstm(X1_test, maxlen=dataset_max_serie_length, value=0.)

	# Load model
	model = create_model(nb_classes=3, img_size=dataset_img_size, max_length=max_length)
	print "Loading model parameters..."
	model.load(models_path+"eyeDNN_HD.tflearn")

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
		train_model()
	elif arg == "test":
		test_model()
	else:
		print "Wrong argument"


