import os
import sys

import os.path
import shutil
import sys
import cv2
import h5py
import random
from random import randrange
import numpy as np

from config import raw_path, dataset_path, dataset_img_size, processed_path, dataset_max_serie_length
from video_tools import get_blank_frame_diff, get_difference_frame

#Padding for LSTM as it appears that padding images is not working in TFLearn
#Not useful if sequences are manually padded (sliding window etc.)
def pad_lstm(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):
	lengths = [len(s) for s in sequences]

	nb_samples = len(sequences)
	if maxlen is None:
		maxlen = np.max(lengths)

	#x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
	x = (np.ones((nb_samples, maxlen, dataset_img_size, dataset_img_size, 1)) * value).astype(dtype)

	for idx, s in enumerate(sequences):
		if len(s) == 0:
			continue  # empty list was found
		if truncating == 'pre':
			trunc = s[-maxlen:]
		elif truncating == 'post':
			trunc = s[:maxlen]
		else:
			raise ValueError("Truncating type '%s' not understood" % padding)

		if padding == 'post':
			x[idx, :len(trunc)] = trunc
		elif padding == 'pre':
			x[idx, -len(trunc):] = trunc
		else:
			raise ValueError("Padding type '%s' not understood" % padding)
	return x

def process(eye):
	"""Processes the image"""
	eye = cv2.resize(eye,(dataset_img_size, dataset_img_size), interpolation = cv2.INTER_CUBIC)
	eye = cv2.equalizeHist(eye)
	return eye

def save(eye, step, eyeNb, motion, destination_path):
	"""Writes image at desired path"""
	name = "{}_{}_{}.png".format(motion,eyeNb,step)
	cv2.imwrite(destination_path + name,eye)

####### GENERATE PROCESSED & AUGMENTED DATASET FROM RAW #######

def generate_processed_imgs():
	"""Reads from RAW and writes to PROCESSED"""

	print("Generating processed images...")

	folders = os.listdir(raw_path)
	folders = [folder for folder in folders if os.path.isdir(raw_path+folder)]

	for index,folder in enumerate(folders):
		os.mkdir(processed_path + folder, exist_ok=True)

		folder_nb = folder.split('_')[-1]
		
		print("Processing folder {}/{}...".format(index+1,len(folders)))
		if index+1 < len(folders): sys.stdout.write("\033[F")

		#Copy skips file
		skip_path = raw_path + folder + '/a_skips.txt'
		newskip_path = processed_path + folder + '/a_skips.txt'
		shutil.copy(skip_path,newskip_path)

		files = os.listdir(raw_path + folder)
		files = [file for file in files if file.endswith(".png")]

		for filename in files:
			motion_name, eyeNb, step = filename[:-4].split('_')
			
			#Open and process image
			eye = cv2.imread(raw_path+folder+"/"+filename,0)
			processedEye = process(eye)
			savePath = processed_path+"{}_{}/".format(motion_name,folder_nb)
			
			#Write processed image on disk
			save(processedEye,step,eyeNb,motion_name,savePath)

####### GENERATE (X,y) DATASET #########

def get_time_warped_dataset(X0, X1, y):
	new_X0, new_X1, new_Y = [], [], []

	#Foreach sample in the dataset
	for i in range(len(y)):
		label = y[i]
		#Grab copie of series
		X0_serie = list(X0[i])
		X1_serie = list(X1[i])

		#Keep initial samples
		new_X0.append(np.array(X0_serie))
		new_X1.append(np.array(X1_serie))
		new_Y.append(label)

		if 25 <= len(X0_serie) <= 49:
			#Slowdown x0.5
			X0_serie = [x for pair in zip(X0_serie, X0_serie) for x in pair]
			X1_serie = [x for pair in zip(X1_serie, X1_serie) for x in pair]
			#Add to dataset
			new_X0.append(np.array(X0_serie))
			new_X1.append(np.array(X1_serie))
			new_Y.append(label)

		elif 74 <= len(X0_serie) <= 100:
			#Speedup x2
			X0_serie = [X0_serie[speed_index] for speed_index in range(len(X0_serie)) if speed_index%2 == 0]
			X1_serie = [X1_serie[speed_index] for speed_index in range(len(X1_serie)) if speed_index%2 == 0]
			#Add to dataset
			new_X0.append(np.array(X0_serie))
			new_X1.append(np.array(X1_serie))
			new_Y.append(label)

	return new_X0, new_X1, new_Y

def get_sliding_padded_dataset(X0, X1, y, missing_paddings):
	"""Pad beginning/end/both to max length"""

	blank_frame = get_blank_frame_diff()
	#We don't keep old samples
	new_X0, new_X1, new_Y = [], [], []

	#Foreach sample in the dataset
	for i in range(len(y)):
		label = y[i]

		missing_frames = dataset_max_serie_length - len(X0[i])
		shift_size = float(missing_frames) / missing_paddings

		#Add N sliding padded examples
		for shift_index in range(missing_paddings + 1):
			#Initialize blank full size series
			X0_serie = [get_blank_frame_diff() for _ in range(dataset_max_serie_length)]
			X1_serie = [get_blank_frame_diff() for _ in range(dataset_max_serie_length)]

			#Change slice of serie
			X0_serie[int(shift_index * shift_size):int(shift_index * shift_size) + len(X0[i])] = X0[i]
			X1_serie[int(shift_index * shift_size):int(shift_index * shift_size) + len(X0[i])] = X1[i]

			#Add to final dataset
			new_X0.append(np.array(X0_serie))
			new_X1.append(np.array(X1_serie))
			new_Y.append(label)

	return new_X0, new_X1, new_Y

def get_random_padded_dataset(X0, X1, y, padded_examples=2):
	blank_frame = get_blank_frame_diff()

	#We don't keep old samples
	new_X0, new_X1, new_Y = [], [], []

	#Foreach sample in the dataset
	for i in range(len(y)):
		label = y[i]

		#Add N padded examples
		for _ in range(padded_examples):
			#Grab copie of series
			X0_serie = list(X0[i])
			X1_serie = list(X1[i])

			#Add blank frames until reaching padded length
			while len(X0_serie) < dataset_max_serie_length:
				random_index = randrange(0, len(X0_serie))
				X0_serie.insert(random_index, blank_frame)
				X1_serie.insert(random_index, blank_frame)

			#Add to final dataset
			new_X0.append(np.array(X0_serie))
			new_X1.append(np.array(X1_serie))
			new_Y.append(label)

	return new_X0, new_X1, new_Y

#Creates pickle/HDF5 dataset (X,y)
def generate_dataset(motions, random_padding=False, cropLength=False, speedUp=False): 
	print("Generating dataset...")

	X0, X1, y = [], [], []

	#List processed folders
	folders = os.listdir(processed_path)
	folders = [folder for folder in folders if os.path.isdir(processed_path + folder)]

	for index, folder in enumerate(folders):

		#Extract ordered stop points
		with open(processed_path + folder + "/a_skips.txt") as f:
			lines = [line.strip() for line in f.readlines()]
			stop_points = [int(line.split('_')[-1]) for line in lines]
			seen = set()
			seen_add = seen.add
			stop_points = [x for x in stop_points if not (x in seen or seen_add(x)) and x != 0]

		print("Extracting data for folder {}/{}...".format(index+1,len(folders)))
		if index + 1 < len(folders): sys.stdout.write("\033[F")

		motion_name, folder_nb = folder.split('_')
		label = [1. if motion_name == motion else 0. for motion in motions]
		
		files = os.listdir(processed_path+folder)
		files = [file for file in files if file.endswith(".png")]

		max_step = max([int(filename[:-4].split('_')[-1]) for filename in files])

		X0_serie, X1_serie = [], []

		for step in range(max_step + 1):
			eye_0_path = processed_path+folder+"/{}_{}_{}.png".format(motion_name,0,step)
			eye_1_path = processed_path+folder+"/{}_{}_{}.png".format(motion_name,1,step)

			#Add first eye image
			eye_0_img = cv2.imread(eye_0_path,0).astype(float)
			eye_0Data = np.reshape(eye_0_img,[dataset_img_size,dataset_img_size,1])
			X0_serie.append(eye_0Data)

			#Add second eye image
			eye_1_img = cv2.imread(eye_1_path,0).astype(float)
			eye_1Data = np.reshape(eye_1_img,[dataset_img_size,dataset_img_size,1])
			X1_serie.append(eye_1Data)

		#Compute frame difference instead of frames
		X0_serie_diff = []
		X1_serie_diff = []

		#Compute diff
		for step in range(1,len(X0_serie)):
			#Avoid shifts of bounding box
			if step not in stop_points:
				eye_0_t0 = X0_serie[step-1]
				eye_0_t1 = X0_serie[step]
				diff0 = get_difference_frame(eye_0_t1, eye_0_t0);
				X0_serie_diff.append(diff0)

				eye_1_t0 = X1_serie[step-1]
				eye_1_t1 = X1_serie[step]
				diff1 = get_difference_frame(eye_1_t1, eye_1_t0);
				X1_serie_diff.append(diff1)

				#Pretty display -- no computation here --
				# from video_tools import show_difference
				# current = np.hstack((eye_0_t1.astype(np.uint8),eye_1_t1.astype(np.uint8)))
				# last = np.hstack((eye_0_t0.astype(np.uint8),eye_1_t0.astype(np.uint8)))
				# show_difference(current,last)
				# cv2.waitKey(0)

		#Add to dataset
		X0.append(np.array(X0_serie_diff))
		X1.append(np.array(X1_serie_diff))
		y.append(label)

	#Now we have loaded all images diffs into X0, X1

	#Manual time warping
	print(len(X0), "examples of shape", X0[0].shape)
	print("Time warping dataset...")
	X0, X1, y = get_time_warped_dataset(X0, X1, y)
	print("Generated dataset of size {} x {}".format(len(X0),X0[0].shape))

	print(len(X0), "examples of shape", X0[0].shape)
	print("Padding dataset (sliding)...")
	X0, X1, y = get_sliding_padded_dataset(X0, X1, y, 8)
	print("Generated dataset of size {} x {}".format(len(X0),X0[0].shape))

	#Shuffle dataset
	data = zip(X0,X1,y)
	random.shuffle(data)
	X0, X1, y = map(list,zip(*data))

	# print("Saving dataset with pickle... - this may take a while")
	# print("Saving X0...")
	# pickle.dump(X0, open(dataset_path+"X0_hd_slide.p", "wb" ))
	# print("Saving X1...")
	# pickle.dump(X1, open(dataset_path+"X1_hd_slide.p", "wb" ))
	# print("Saving y...")
	# pickle.dump(y, open(dataset_path+"y_hd_slide.p", "wb" ))

	#Save at hdf5 format
	print("Saving dataset in hdf5 format... - this may take a while")
	h5f = h5py.File(dataset_path+"data.h5", 'w')
	h5f.create_dataset('eye_X0', data=X0)
	h5f.create_dataset('eye_X1', data=X1)
	h5f.create_dataset('eye_Y', data=y)
	h5f.close()

if __name__ == "__main__":
	motions = ["gamma","z","idle"]

	arg = sys.argv[1] if len(sys.argv) == 2 else None 

	if arg is None:
		print "Need argument"
	elif arg == "process":
		generate_processed_imgs()
	elif arg == "dataset":
		generate_dataset(motions, random_padding=True, cropLength=False, speedUp=False)
	else:
		print "Wrong argument"



