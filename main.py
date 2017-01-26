import cv2
import numpy as np
import tflearn
from model import createModel
from config import datasetImageSize
from config import framesInHistory
from datasetTools import padLSTM
from datasetTools import process
from imutils.video import WebcamVideoStream
import cv2
import imutils
import time
from detector import Detector
from videoTools import displayCurrentDiff
from videoTools import displayHistoryDiffs
from videoTools import getBlankFrameDiff
from videoTools import getDifferenceFrame
from videoTools import showDifference
from classifier import Classifier

def main(displayHistory=True):
	#Window for past frames
	framesDiffHistory = [(getBlankFrameDiff(),getBlankFrameDiff()) for i in range(framesInHistory)]
	lastEyes = None

	#Load model classifier
	classifier = Classifier()
	#Start thread to make predictions
	classifier.startPredictions()

	#Initialize webcam
	vs = WebcamVideoStream(src=0).start()
	
	#For FPS computation
	t0 = -1

	#Face/eyes detector
	detector = Detector()

	print "Starting eye recognition..."
	while True:

		#Compute FPS
		dt =  time.time() - t0
		fps = 1/dt
		t0 = time.time()

		#Limit FPS with wait
		waitMs = 5
		key = cv2.waitKey(waitMs) & 0xFF

		#Get image from webcam, convert to grayscale and resize
		fullFrame = vs.read()
		fullFrame = cv2.cvtColor(fullFrame, cv2.COLOR_BGR2GRAY)
		frame = imutils.resize(fullFrame, width=300)

		#Find face
		faceBB = detector.getFace(frame)
		if faceBB is None:
			#Invalidate eyes bounding box as all will change
			lastEyes = None
			detector.resetEyesBB()
			continue

		#Get low resolution face coordinates
		x,y,w,h = faceBB
		face = frame[y:y+h, x:x+w]

		#Apply to high resolution frame
		xScale = fullFrame.shape[1]/frame.shape[1]
		yScale = fullFrame.shape[0]/frame.shape[0]
		x,y,w,h = x*xScale,y*yScale,w*xScale,h*yScale
		fullFace = fullFrame[y:y+h, x:x+w]

		#Find eyes on high resolution face
		eyes = detector.getEyes(fullFace)
		if eyes is None:
			#Reset last eyes
			lastEyes = None
			continue

		eye0, eye1 = eyes

		#Process (normalize, resize)			
		eye0 = process(eye0)
		eye1 = process(eye1)
		
		#Reshape for dataset
		eye0 = np.reshape(eye0,[datasetImageSize,datasetImageSize,1])
		eye1 = np.reshape(eye1,[datasetImageSize,datasetImageSize,1])

		#We have a recent picture of the eyes
		if lastEyes is not None:
			#Load previous eyes
			eye0previous, eye1previous = lastEyes

			#Compute diffs
			diff0 = getDifferenceFrame(eye0, eye0previous)
			diff1 = getDifferenceFrame(eye1, eye1previous)

			#Display/debug
			displayDiff = False
			if displayDiff:
				displayCurrentDiff(eye0,eye1,eye0previous,eye1previous, stopFrame=False)

			#Crop beginning then add new to end
			framesDiffHistory = framesDiffHistory[1:]
			framesDiffHistory.append([diff0,diff1])

		#Keep current as last frame
		lastEyes = [eye0, eye1]

		#Note: this is not time consuming
		if displayHistory:
			displayHistoryDiffs(framesDiffHistory, fps)

		#Extract each eyes
		X0, X1 = zip(*framesDiffHistory)

		#Reshape as a tensor (NbExamples,SerieLength,Width,Height,Channels)
		X0 = np.reshape(X0,[-1,len(framesDiffHistory),datasetImageSize,datasetImageSize,1])
		X1 = np.reshape(X1,[-1,len(framesDiffHistory),datasetImageSize,datasetImageSize,1])

		#Save history to Classifier
		classifier.X0 = X0
		classifier.X1 = X1

		#TODO custom actions (change volume, send notification, close app etc.)
		#Handle verified patterns from history
		#detectPattern(classifier)	


if __name__ == "__main__":
	main(displayHistory=True)








