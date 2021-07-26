import pickle
from config import dataset_path
from dataset_tools import pad_lstm
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#Useless if done on padded dataset
def explore():
	#Load dataset
	X0 = pickle.load(open(dataset_path+"X0_hd.p", "rb" ))
	X1 = pickle.load(open(dataset_path+"X1_hd.p", "rb" ))
	y = pickle.load(open(dataset_path+"y_hd.p", "rb" ))

	lengths = [serie.shape[0] for serie in X0]
	min_length, max_length = min(lengths), max(lengths)

	#print lengths
	print len([_ for _ in lengths if _ > 100]), "over 100"
	print "Lengths range: {} to {}".format(min_length,max_length)
	print len(X0), "series of shape", X0[0].shape

	X0 = pad_lstm(X0, maxlen=max_length, value=0.)
	X1 = pad_lstm(X1, maxlen=max_length, value=0.)

	print len(X0), "padded series of shape", X0[0].shape

	# the histogram of the data
	n, bins, patches = plt.hist(lengths, 100, facecolor='blue', alpha=0.75)
	plt.show()


if __name__ == "__main__":
	explore()



