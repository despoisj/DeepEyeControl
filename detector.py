import cv2
from config import cv2_cascade_path
import math

class Detector:

	def __init__(self):
		self.face_cascade = cv2.CascadeClassifier(cv2_cascade_path + 'haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier(cv2_cascade_path + 'haarcascade_eye.xml')
		self.face_bb = None
		self.eyes_bb = {'left':None, 'right':None}

	def reset_eyes_bb(self):
		self.eyes_bb = {'left':None, 'right':None}

	def is_same_face_bb(self,bb1,bb2):
		x1,y1,w1,h1 = bb1
		x2,y2,w2,h2 = bb2
		tolerance = 5
		return math.fabs(w1-w2) < tolerance and math.fabs(h1-h2) < tolerance and math.fabs(x1-x2) < tolerance and math.fabs(y1-y2) < tolerance

	def is_same_eye_bb(self,bb1,bb2):
		x1,y1,w1,h1 = bb1
		x2,y2,w2,h2 = bb2
		toleranceX = 10
		toleranceY = 10
		toleranceW = 10
		toleranceH = 10

		return math.fabs(w1-w2) < toleranceW and math.fabs(h1-h2) < toleranceH and math.fabs(x1-x2) < toleranceX and math.fabs(y1-y2) < toleranceY

	def get_face(self, frame):
		#Find faces
		faces = self.face_cascade.detectMultiScale(
			frame,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(50, 50),
			maxSize=(110, 110),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)

		#Skip frame if no face
		if len(faces) != 1:
			return None
		
		face_bb = faces[0]

		#Save face for first frame
		if self.face_bb is None:
			self.face_bb = face_bb
		#Check if similar bounding box
		elif self.is_same_face_bb(face_bb,self.face_bb):
			#Same-ish BB, load
			face_bb = self.face_bb
		else:
			#New BB, save and skip frame
			self.face_bb = face_bb
			return None

		return face_bb

	#Returns left eye then right (on picture)
	def get_eyes(self, face):
		
		#Find eyes in the face
		eyes = self.eye_cascade.detectMultiScale(
			face,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(60, 60),
			maxSize=(100, 100),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)

		if len(eyes) != 2:
			return None

		left_eyeX = 999
		left_eye = None
		right_eye = None

		#Find left and right eyes
		for ex,ey,ew,eh in eyes:
			#New left eye
			if ex < left_eyeX:
				if left_eye is not None:
					right_eye = left_eye
					left_eye = (ex,ey,ew,eh)
				else:
					left_eye = (ex,ey,ew,eh)
					left_eyeX = ex
			else:
				if left_eye is not None:
					right_eye = (ex,ey,ew,eh)

		#Stabilization
		eyes_bb = {'left':left_eye,'right':right_eye}

		for side in ['left','right']:

			#Save first frame BB
			if self.eyes_bb[side] is None:
				self.eyes_bb[side] = eyes_bb[side]
			#Load if similar BB
			elif self.is_same_eye_bb(eyes_bb[side],self.eyes_bb[side]):
				eyes_bb[side] = self.eyes_bb[side]
			#Changed the Bounding Box
			else:
				#New BB, save and skip frame
				self.eyes_bb[side] = eyes_bb[side] 
				return None
			
		#Get BB for cropping
		xLeft, yLeft, wLeft, hLeft = eyes_bb['left']
		xRight, yRight, wRight, hRight = eyes_bb['right']

		focus_on_center = False

		if focus_on_center:
			# #Focus on the center of the eye
			wLeftNew = int(wLeft*0.75)
			hLeftNew = int(hLeft*0.75)
			xLeftNew = int(xLeft+float(wLeft)*0.5-float(wLeftNew)*0.5)
			yLeftNew = int(yLeft+float(hLeft)*0.55-float(hLeftNew)*0.45)

			xLeft, yLeft, wLeft, hLeft = xLeftNew, yLeftNew, wLeftNew, hLeftNew

			# #Focus on the center of the eye
			wRightNew = int(wRight*0.75)
			hRightNew = int(hRight*0.75)
			xRightNew = int(xRight+float(wRight)*0.5-float(wRightNew)*0.5)
			yRightNew = int(yRight+float(hRight)*0.55-float(hRightNew)*0.45)

			xRight, yRight, wRight, hRight = xRightNew, yRightNew, wRightNew, hRightNew

		left_eyeImage = face[yLeft:yLeft+hLeft, xLeft:xLeft+wLeft]
		right_eyeImage = face[yRight:yRight+hRight, xRight:xRight+wRight]

		return left_eyeImage, right_eyeImage
