import os
import time

import cv2
import numpy as np

import imutils
from imutils.video import WebcamVideoStream

from config import rawPath
from datasetTools import save, process
from videoTools import getEyes
from videoTools import getFace
from videoTools import showDifference

from detector import Detector


#Used to show/save NON-PROCESSED images in "raw"
def stream(record, showFrame, showFace, showEyes, showDiff):
	
	#Setup motion recording
	motionIndex = 0
	folders = os.listdir(rawPath)
	for folder in folders:
		if folder.startswith(motionName):
			number = int(folder.split("_")[-1])
			motionIndex = max(motionIndex,number + 1)

	print "Recording {}_{}".format(motionName,motionIndex)
	savePath = rawPath+"{}_{}/".format(motionName,motionIndex)
	
	if record:
		os.mkdir(savePath)

	detector = Detector()

	#Video loop
	vs = WebcamVideoStream(src=0).start()
	
	dt = 0
	step = 0

	lastEye0 = None
	lastFace = None

	fullFrameSize = (1280,720)

	with open(savePath+'/a_skips.txt','w+') as f:

		while True:	
			
			t0 = time.time()

			waitMs = 5
			key = cv2.waitKey(waitMs) & 0xFF	

			fullFrame = vs.read()
			fullFrame = cv2.cvtColor(fullFrame, cv2.COLOR_BGR2GRAY)
			frame = imutils.resize(fullFrame, width=300)

			#Show frame
			if showFrame:
				cv2.imshow("Frame", imutils.resize(frame, width=300))

			faceBB = detector.getFace(frame)

			#Skip if no face
			if faceBB is None:
				#Avoid flash in difference
				lastFace = None
				#Invalidate eyes bounding box as all will change
				detector.resetEyesBB();
				#Write skips for difference frames
				f.write("face_{}\n".format(step))
				continue

			#Get small face coordinates
			x,y,w,h = faceBB
			face = frame[y:y+h, x:x+w]

			#Apply to fullscale
			xScale = fullFrame.shape[1]/frame.shape[1]
			yScale = fullFrame.shape[0]/frame.shape[0]
			x,y,w,h = x*xScale,y*yScale,w*xScale,h*yScale
			fullFace = fullFrame[y:y+h, x:x+w]

			#Show face
			if showFace:
				if lastFace is not None and showDiff:
					showDifference(fullFace,lastFace)
				else:
					resizedFullFace = cv2.resize(fullFace,(200,200))
					resizedFace = cv2.resize(face,(200,200))
					cv2.imshow("Face",np.hstack((resizedFullFace,resizedFace)))

			#Remember last face
			lastFace = fullFace

			#Get eyes
			eyes = detector.getEyes(fullFace)

			#Skip if not 2 eyes
			if eyes is None:
				lastEye0 = None
				#Write skips for difference frames
				f.write("eye_{}\n".format(step))
				continue

			eye0, eye1 = eyes
			
			if showEyes:
				if showDiff and lastEye0 is not None:
					showDifference(eye0,lastEye0)
				else:	
					processed0 = cv2.resize(process(eye0),(100,100))
					processed1 = cv2.resize(process(eye1),(100,100))
					processed = np.hstack((processed0,processed1))
					cv2.imshow("Eyes",processed)

				lastEye0 = eye0

			if record:
				save(eye0,step,0,motionName,savePath)
				save(eye1,step,1,motionName,savePath)
				step += 1

			dt += time.time() - t0
			#print "FPS: {}".format(int(i/dt))


if __name__ == "__main__":

	motionName = "z"
	stream(record=True, showFrame=False, showFace=False, showEyes=False, showDiff=False)











