import os
import time

import cv2
import numpy as np
import imutils
from imutils.video import WebcamVideoStream

from config import raw_path
from dataset_tools import save, process
from video_tools import get_eyes, get_face, show_difference
from detector import Detector


def stream(record, show_frame, show_face, show_eyes, show_diff):
	"""Used to show/save NON-PROCESSED images in 'raw'"""

	#Setup motion recording
	motion_index = 0
	folders = os.listdir(raw_path)
	for folder in folders:
		if folder.startswith(motion_name):
			number = int(folder.split("_")[-1])
			motion_index = max(motion_index, number + 1)

	print("Recording {}_{}".format(motion_name, motion_index))
	save_path = raw_path+"{}_{}/".format(motion_name, motion_index)
	
	if record:
		os.mkdir(save_path)

	detector = Detector()

	#Video loop
	vs = WebcamVideoStream(src=0).start()
	
	dt = 0
	step = 0

	lasteye_0 = None
	last_face = None

	full_frameSize = (1280,720)

	with open(save_path + '/a_skips.txt','w+') as f:

		while True:	
			
			t0 = time.time()

			waitMs = 5
			key = cv2.waitKey(waitMs) & 0xFF	

			full_frame = vs.read()
			full_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
			frame = imutils.resize(full_frame, width=300)

			#Show frame
			if show_frame:
				cv2.imshow("Frame", imutils.resize(frame, width=300))

			face_bb = detector.get_face(frame)

			#Skip if no face
			if face_bb is None:
				#Avoid flash in difference
				last_face = None
				#Invalidate eyes bounding box as all will change
				detector.reset_eyes_bb();
				#Write skips for difference frames
				f.write("face_{}\n".format(step))
				continue

			#Get small face coordinates
			x,y,w,h = face_bb
			face = frame[y:y+h, x:x+w]

			#Apply to fullscale
			xScale = full_frame.shape[1] / frame.shape[1]
			yScale = full_frame.shape[0] / frame.shape[0]
			x,y,w,h = x * xScale, y * yScale, w * xScale, h * yScale
			fullFace = full_frame[y:y+h, x:x+w]

			#Show face
			if show_face:
				if last_face is not None and show_diff:
					show_difference(fullFace,last_face)
				else:
					resizedFullFace = cv2.resize(fullFace,(200,200))
					resizedFace = cv2.resize(face,(200,200))
					cv2.imshow("Face",np.hstack((resizedFullFace,resizedFace)))

			#Remember last face
			last_face = fullFace

			#Get eyes
			eyes = detector.get_eyes(fullFace)

			#Skip if not 2 eyes
			if eyes is None:
				lasteye_0 = None
				#Write skips for difference frames
				f.write("eye_{}\n".format(step))
				continue

			eye_0, eye_1 = eyes
			
			if show_eyes:
				if show_diff and lasteye_0 is not None:
					show_difference(eye_0,lasteye_0)
				else:	
					processed0 = cv2.resize(process(eye_0),(100,100))
					processed1 = cv2.resize(process(eye_1),(100,100))
					processed = np.hstack((processed0,processed1))
					cv2.imshow("Eyes",processed)

				lasteye_0 = eye_0

			if record:
				save(eye_0,step,0,motion_name,save_path)
				save(eye_1,step,1,motion_name,save_path)
				step += 1

			dt += time.time() - t0
			#print "FPS: {}".format(int(i/dt))


if __name__ == "__main__":

	motion_name = "z"
	stream(record=True, show_frame=False, show_face=False, show_eyes=False, show_diff=False)











