import cv2
import numpy as np
from config import dataset_img_size
import math

# Show current eyes with diff
def display_current_diff(eye_0, eye_1, eye_0_previous, eye_1_previous, stop_frame=False):
	current = np.hstack((eye_0.astype(np.uint8),eye_1.astype(np.uint8)))
	last = np.hstack((eye_0_previous.astype(np.uint8),eye_1_previous.astype(np.uint8)))
	show_difference(current,last)
	
	# Debug frame by frame
	if stop_frame:
		cv2.waitKey(0)
		
#Display whole history as seen by LSTM
def display_history_diffs(frames_diff_history, fps):
	frames_diff_history_imgs = []
	for diff, _ in frames_diff_history:
		frames_diff_history_imgs.append(get_showable(diff))

	img = None
	#Size of display square. Ex. 64 -> 8
	square_size = int(math.sqrt(len(frames_diff_history)))
	for row_index in range(square_size):
		#Debug history
		row_img = np.hstack(frames_diff_history_imgs[row_index*square_size:row_index*square_size+square_size])
		img = np.vstack((img,row_img)) if img is not None else row_img
	
	#Write FPS on image
	cv2.putText(img,"FPS: {}".format(int(fps)),(3,9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
	cv2.imshow("Frame history",img)

# Returns blank frame difference in float numpy array
def get_blank_frame_diff():
	return np.zeros((dataset_img_size, dataset_img_size, 1))

# Returns difference frame in float numpy array
def get_difference_frame(t1, t0):
	diff = t1.astype(float) - t0.astype(float)
	return diff

# Computes difference and returns it in a 
# format good for OpenCV imshow
def get_showable_difference(t1, t0):
	diff = get_difference_frame(t1,t0)
	return get_showable(diff)

# Converts float diff to good OpenCV format for display
def get_showable(diff):
	diff = (diff+255.)/2.
	return diff.astype(np.uint8)

# Displays difference between frames
def show_difference(t1,t0):
	#Resize for consistency
	t0 = cv2.resize(t0,(400,400))
	t1 = cv2.resize(t1,(400,400))

	#Get displayable diff
	diff = get_showable_difference(t1,t0)

	#Resize to show better
	t0 = cv2.resize(t0,(100,100))
	t1 = cv2.resize(t1,(100,100))
	diff = cv2.resize(diff,(100,100))

	#Display
	img = np.hstack((t0,t1,diff))
	t0 = cv2.equalizeHist(t0)
	t1 = cv2.equalizeHist(t1)

	#Get displayable diff
	diff = get_showable_difference(t1,t0)

	#Resize to show better
	t0 = cv2.resize(t0,(100,100))
	t1 = cv2.resize(t1,(100,100))
	diff = cv2.resize(diff,(100,100))

	#Display
	processed_img = np.hstack((t0,t1,diff))

	cv2.imshow("Difference", np.vstack((img,processed_img)))
