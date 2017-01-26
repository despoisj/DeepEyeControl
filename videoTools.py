import cv2
import numpy as np
from config import datasetImageSize
import math

# Show current eyes with diff
def displayCurrentDiff(eye0, eye1, eye0previous, eye1previous, stopFrame=False):
	current = np.hstack((eye0.astype(np.uint8),eye1.astype(np.uint8)))
	last = np.hstack((eye0previous.astype(np.uint8),eye1previous.astype(np.uint8)))
	showDifference(current,last)
	
	# Debug frame by frame
	if stopFrame:
		cv2.waitKey(0)
		
#Display whole history as seen by LSTM
def displayHistoryDiffs(framesDiffHistory, fps):
	framesDiffHistoryImages = []
	for diff, _ in framesDiffHistory:
		framesDiffHistoryImages.append(getShowable(diff))

	img = None
	#Size of display square. Ex. 64 -> 8
	squareSize = int(math.sqrt(len(framesDiffHistory)))
	for rowIndex in range(squareSize):
		#Debug history
		rowImg = np.hstack(framesDiffHistoryImages[rowIndex*squareSize:rowIndex*squareSize+squareSize])
		img = np.vstack((img,rowImg)) if img is not None else rowImg
	
	#Write FPS on image
	cv2.putText(img,"FPS: {}".format(int(fps)),(3,9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
	cv2.imshow("Frame history",img)

# Returns blank frame difference in float numpy array
def getBlankFrameDiff():
	return np.zeros((datasetImageSize,datasetImageSize)).reshape([datasetImageSize,datasetImageSize,1])

# Returns difference frame in float numpy array
def getDifferenceFrame(t1, t0):
	diff = t1.astype(float) - t0.astype(float)
	return diff

# Computes difference and returns it in a 
# format good for OpenCV imshow
def getShowableDifference(t1, t0):
	diff = getDifferenceFrame(t1,t0)
	return getShowable(diff)

# Converts float diff to good OpenCV format for display
def getShowable(diff):
	diff = (diff+255.)/2.
	return diff.astype(np.uint8)

# Displays difference between frames
def showDifference(t1,t0):
	#Resize for consistency
	t0 = cv2.resize(t0,(400,400))
	t1 = cv2.resize(t1,(400,400))

	#Get displayable diff
	diff = getShowableDifference(t1,t0)

	#Resize to show better
	t0 = cv2.resize(t0,(100,100))
	t1 = cv2.resize(t1,(100,100))
	diff = cv2.resize(diff,(100,100))

	#Display
	img = np.hstack((t0,t1,diff))
	t0 = cv2.equalizeHist(t0)
	t1 = cv2.equalizeHist(t1)

	#Get displayable diff
	diff = getShowableDifference(t1,t0)

	#Resize to show better
	t0 = cv2.resize(t0,(100,100))
	t1 = cv2.resize(t1,(100,100))
	diff = cv2.resize(diff,(100,100))

	#Display
	processedImg = np.hstack((t0,t1,diff))

	cv2.imshow("Difference", np.vstack((img,processedImg)))
