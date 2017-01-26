import os
import cv2
import random
from random import randrange
import numpy as np
from config import rawPath, datasetPath, datasetImageSize, processedPath, datasetMaxSerieLength
import os.path
import shutil
import sys
from videoTools import getBlankFrameDiff, getDifferenceFrame

#Padding for LSTM as it appears that padding images is not working in TFLearn
#Not useful if sequences are manually padded (sliding window etc.)
def padLSTM(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):
	lengths = [len(s) for s in sequences]

	nb_samples = len(sequences)
	if maxlen is None:
		maxlen = np.max(lengths)

	#x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
	x = (np.ones((nb_samples, maxlen, datasetImageSize, datasetImageSize, 1)) * value).astype(dtype)

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

#All processing on the image
def process(eye):
	eye = cv2.resize(eye,(datasetImageSize, datasetImageSize), interpolation = cv2.INTER_CUBIC)
	eye = cv2.equalizeHist(eye)
	return eye

#Writes image at desired path
def save(eye,step,eyeNb,motion,destinationPath):
	name = "{}_{}_{}.png".format(motion,eyeNb,step)
	cv2.imwrite(destinationPath+name,eye)

####### GENERATE PROCESSED & AUGMENTED DATASET FROM RAW #######

#Reads from RAW and writes to PROCESSED
def generateProcessedImages():
	print "Generating processed images..."

	folders = os.listdir(rawPath)
	folders = [folder for folder in folders if os.path.isdir(rawPath+folder)]

	for index,folder in enumerate(folders):
		try:
			os.mkdir(processedPath+folder)
		except:
			pass

		folderNb = folder.split('_')[-1]
		
		print "Processing folder {}/{}...".format(index+1,len(folders))
		if index+1 < len(folders): sys.stdout.write("\033[F")

		#Copy skips file
		skipPath = rawPath+folder+'/a_skips.txt'
		newSkipPath = processedPath+folder+'/a_skips.txt'
		shutil.copy(skipPath,newSkipPath)

		files = os.listdir(rawPath+folder)
		files = [file for file in files if file.endswith(".png")]

		for filename in files:
			motionName, eyeNb, step = filename[:-4].split('_')
			
			#Open and process image
			eye = cv2.imread(rawPath+folder+"/"+filename,0)
			processedEye = process(eye)
			savePath = processedPath+"{}_{}/".format(motionName,folderNb)
			
			#Write processed image on disk
			save(processedEye,step,eyeNb,motionName,savePath)

####### GENERATE (X,y) DATASET #########

def getTimeWarpedDataset(X0,X1,y):

	newX0, newX1, newY = [], [], []

	#Foreach sample in the dataset
	for i in range(len(y)):
		label = y[i]
		#Grab copie of series
		X0serie = list(X0[i])
		X1serie = list(X1[i])

		#Keep initial samples
		newX0.append(np.array(X0serie))
		newX1.append(np.array(X1serie))
		newY.append(label)

		if 25 <= len(X0serie) <= 49:
			#Slowdown x0.5
			X0serie = [x for pair in zip(X0serie,X0serie) for x in pair]
			X1serie = [x for pair in zip(X1serie,X1serie) for x in pair]
			#Add to dataset
			newX0.append(np.array(X0serie))
			newX1.append(np.array(X1serie))
			newY.append(label)

		elif 74 <= len(X0serie) <= 100:
			#Speedup x2
			X0serie = [X0serie[speedIndex] for speedIndex in range(len(X0serie)) if speedIndex%2 == 0]
			X1serie = [X1serie[speedIndex] for speedIndex in range(len(X1serie)) if speedIndex%2 == 0]
			#Add to dataset
			newX0.append(np.array(X0serie))
			newX1.append(np.array(X1serie))
			newY.append(label)

	return newX0, newX1, newY

#Pad beginning/end/both to max length
def getSlidingPaddedDataset(X0,X1,y, missingPaddings):
	blankFrame = getBlankFrameDiff()
	#We don't keep old samples
	newX0, newX1, newY = [], [], []

	#Foreach sample in the dataset
	for i in range(len(y)):
		label = y[i]

		missingFrames = datasetMaxSerieLength - len(X0[i])
		shiftSize = float(missingFrames)/missingPaddings

		#Add N sliding padded examples
		for shiftIndex in range(missingPaddings+1):
			#Initialize blank full size series
			X0serie = [getBlankFrameDiff() for _ in range(datasetMaxSerieLength)]
			X1serie = [getBlankFrameDiff() for _ in range(datasetMaxSerieLength)]

			#Change slice of serie
			X0serie[int(shiftIndex*shiftSize):int(shiftIndex*shiftSize)+len(X0[i])] = X0[i]
			X1serie[int(shiftIndex*shiftSize):int(shiftIndex*shiftSize)+len(X0[i])] = X1[i]

			#Add to final dataset
			newX0.append(np.array(X0serie))
			newX1.append(np.array(X1serie))
			newY.append(label)

	return newX0, newX1, newY

def getRandomPaddedDataset(X0,X1,y,paddedExamples=2):
	blankFrame = getBlankFrameDiff()

	#We don't keep old samples
	newX0, newX1, newY = [], [], []

	#Foreach sample in the dataset
	for i in range(len(y)):
		label = y[i]

		#Add N padded examples
		for _ in range(paddedExamples):
			#Grab copie of series
			X0serie = list(X0[i])
			X1serie = list(X1[i])

			#Add blank frames until reaching padded length
			while len(X0serie) < datasetMaxSerieLength:
				randomIndex = randrange(0,len(X0serie))
				X0serie.insert(randomIndex,blankFrame)
				X1serie.insert(randomIndex,blankFrame)

			#Add to final dataset
			newX0.append(np.array(X0serie))
			newX1.append(np.array(X1serie))
			newY.append(label)

	return newX0, newX1, newY

#Creates pickle/HDF5 dataset (X,y)
def generateDataset(motions, randomPadding=False, cropLength=False, speedUp=False): 
	print "Generating dataset..."

	X0, X1, y = [], [], []

	#List processed folders
	folders = os.listdir(processedPath)
	folders = [folder for folder in folders if os.path.isdir(processedPath+folder)]

	for index, folder in enumerate(folders):

		#Extract ordered stop points
		with open(processedPath+folder+"/a_skips.txt") as f:
			lines = [line.strip() for line in f.readlines()]
			stopPoints = [int(line.split('_')[-1]) for line in lines]
			seen = set()
			seen_add = seen.add
			stopPoints = [x for x in stopPoints if not (x in seen or seen_add(x)) and x != 0]

		print "Extracting data for folder {}/{}...".format(index+1,len(folders))
		if index+1 < len(folders): sys.stdout.write("\033[F")

		motionName, folderNb = folder.split('_')
		label = [1. if motionName == motion else 0. for motion in motions]
		
		files = os.listdir(processedPath+folder)
		files = [file for file in files if file.endswith(".png")]

		maxStep = max([int(filename[:-4].split('_')[-1]) for filename in files])

		X0serie, X1serie = [], []

		for step in range(maxStep+1):
			eye0Path = processedPath+folder+"/{}_{}_{}.png".format(motionName,0,step)
			eye1Path = processedPath+folder+"/{}_{}_{}.png".format(motionName,1,step)

			#Add first eye image
			eye0Img = cv2.imread(eye0Path,0).astype(float)
			eye0Data = np.reshape(eye0Img,[datasetImageSize,datasetImageSize,1])
			X0serie.append(eye0Data)

			#Add second eye image
			eye1Img = cv2.imread(eye1Path,0).astype(float)
			eye1Data = np.reshape(eye1Img,[datasetImageSize,datasetImageSize,1])
			X1serie.append(eye1Data)

		#Compute frame difference instead of frames
		X0serieDiff = []
		X1serieDiff = []

		#Compute diff
		for step in range(1,len(X0serie)):
			#Avoid shifts of bounding box
			if step not in stopPoints:
				eye0_t0 = X0serie[step-1]
				eye0_t1 = X0serie[step]
				diff0 = getDifferenceFrame(eye0_t1, eye0_t0);
				X0serieDiff.append(diff0)

				eye1_t0 = X1serie[step-1]
				eye1_t1 = X1serie[step]
				diff1 = getDifferenceFrame(eye1_t1, eye1_t0);
				X1serieDiff.append(diff1)

				#Pretty display -- no computation here --
				# from videoTools import showDifference
				# current = np.hstack((eye0_t1.astype(np.uint8),eye1_t1.astype(np.uint8)))
				# last = np.hstack((eye0_t0.astype(np.uint8),eye1_t0.astype(np.uint8)))
				# showDifference(current,last)
				# cv2.waitKey(0)

		#Add to dataset
		X0.append(np.array(X0serieDiff))
		X1.append(np.array(X1serieDiff))
		y.append(label)

	#Now we have loaded all images diffs into X0, X1

	#Manual time warping
	print len(X0), "examples of shape", X0[0].shape
	print "Time warping dataset..."
	X0, X1, y = getTimeWarpedDataset(X0, X1, y)
	print "Generated dataset of size {} x {}".format(len(X0),X0[0].shape)

	#print len(X0), "examples of shape", X0[0].shape
	#print "Padding dataset (random)..."
	#X0, X1, y = getRandomPaddedDataset(X0, X1, y, 2)
	#print "Generated dataset of size {} x {}".format(len(X0),X0[0].shape)

	print len(X0), "examples of shape", X0[0].shape
	print "Padding dataset (sliding)..."
	X0, X1, y = getSlidingPaddedDataset(X0, X1, y, 8)
	print "Generated dataset of size {} x {}".format(len(X0),X0[0].shape)

	#Shuffle dataset
	data = zip(X0,X1,y)
	random.shuffle(data)
	X0, X1, y = map(list,zip(*data))

	# print "Saving dataset with pickle... - this may take a while"
	# print "Saving X0..."
	# pickle.dump(X0, open(datasetPath+"X0_hd_slide.p", "wb" ))
	# print "Saving X1..."
	# pickle.dump(X1, open(datasetPath+"X1_hd_slide.p", "wb" ))
	# print "Saving y..."
	# pickle.dump(y, open(datasetPath+"y_hd_slide.p", "wb" ))

	#Save at hdf5 format
	print "Saving dataset in hdf5 format... - this may take a while"
	import h5py
	h5f = h5py.File(datasetPath+"data.h5", 'w')
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
		generateProcessedImages()
	elif arg == "dataset":
		generateDataset(motions, randomPadding=True, cropLength=False, speedUp=False)
	else:
		print "Wrong argument"



