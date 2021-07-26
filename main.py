import cv2
import numpy as np
import tflearn
import cv2
import imutils
import time
from imutils.video import WebcamVideoStream

from model import create_model
from config import dataset_img_size
from config import frames_in_history
from dataset_tools import pad_lstm, process
from video_tools import display_current_diff, display_history_diffs, get_blank_frame_diff, get_difference_frame, show_difference
from detector import Detector
from classifier import Classifier

def main(display_history=True):
	#Window for past frames
	frames_diff_history = [(get_blank_frame_diff(),get_blank_frame_diff()) for i in range(frames_in_history)]
	last_eyes = None

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
		full_frame = vs.read()
		full_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
		frame = imutils.resize(full_frame, width=300)

		#Find face
		face_bb = detector.get_face(frame)
		if face_bb is None:
			#Invalidate eyes bounding box as all will change
			last_eyes = None
			detector.reset_eyes_bb()
			continue

		#Get low resolution face coordinates
		x,y,w,h = face_bb
		face = frame[y:y+h, x:x+w]

		#Apply to high resolution frame
		xScale = full_frame.shape[1]/frame.shape[1]
		yScale = full_frame.shape[0]/frame.shape[0]
		x,y,w,h = x*xScale,y*yScale,w*xScale,h*yScale
		fullFace = full_frame[y:y+h, x:x+w]

		#Find eyes on high resolution face
		eyes = detector.get_eyes(fullFace)
		if eyes is None:
			#Reset last eyes
			last_eyes = None
			continue

		eye_0, eye_1 = eyes

		#Process (normalize, resize)			
		eye_0 = process(eye_0)
		eye_1 = process(eye_1)
		
		#Reshape for dataset
		eye_0 = np.reshape(eye_0,[dataset_img_size,dataset_img_size,1])
		eye_1 = np.reshape(eye_1,[dataset_img_size,dataset_img_size,1])

		#We have a recent picture of the eyes
		if last_eyes is not None:
			#Load previous eyes
			eye_0_previous, eye_1_previous = last_eyes

			#Compute diffs
			diff0 = get_difference_frame(eye_0, eye_0_previous)
			diff1 = get_difference_frame(eye_1, eye_1_previous)

			#Display/debug
			displayDiff = False
			if displayDiff:
				display_current_diff(eye_0,eye_1,eye_0_previous,eye_1_previous, stop_frame=False)

			#Crop beginning then add new to end
			frames_diff_history = frames_diff_history[1:]
			frames_diff_history.append([diff0,diff1])

		#Keep current as last frame
		last_eyes = [eye_0, eye_1]

		#Note: this is not time consuming
		if display_history:
			display_history_diffs(frames_diff_history, fps)

		#Extract each eyes
		X0, X1 = zip(*frames_diff_history)

		#Reshape as a tensor (NbExamples,SerieLength,Width,Height,Channels)
		X0 = np.reshape(X0,[-1,len(frames_diff_history),dataset_img_size,dataset_img_size,1])
		X1 = np.reshape(X1,[-1,len(frames_diff_history),dataset_img_size,dataset_img_size,1])

		#Save history to Classifier
		classifier.X0 = X0
		classifier.X1 = X1

		#TODO custom actions (change volume, send notification, close app etc.)
		#Handle verified patterns from history
		#detectPattern(classifier)	


if __name__ == "__main__":
	main(display_history=True)
