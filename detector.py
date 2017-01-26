import cv2
from config import cascadePath
import math

class Detector:

	def __init__(self):
		self.face_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier(cascadePath+'haarcascade_eye.xml')
		self.faceBB = None
		self.eyesBB = {'left':None, 'right':None}

	def resetEyesBB(self):
		self.eyesBB = {'left':None, 'right':None}

	def isSameFaceBB(self,bb1,bb2):
		x1,y1,w1,h1 = bb1
		x2,y2,w2,h2 = bb2
		tolerance = 5
		return math.fabs(w1-w2) < tolerance and math.fabs(h1-h2) < tolerance and math.fabs(x1-x2) < tolerance and math.fabs(y1-y2) < tolerance

	def isSameEyeBB(self,bb1,bb2):
		x1,y1,w1,h1 = bb1
		x2,y2,w2,h2 = bb2
		toleranceX = 10
		toleranceY = 10
		toleranceW = 10
		toleranceH = 10

		return math.fabs(w1-w2) < toleranceW and math.fabs(h1-h2) < toleranceH and math.fabs(x1-x2) < toleranceX and math.fabs(y1-y2) < toleranceY

	def getFace(self, frame):
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
		
		faceBB = faces[0]

		#Save face for first frame
		if self.faceBB is None:
			self.faceBB = faceBB
		#Check if similar bounding box
		elif self.isSameFaceBB(faceBB,self.faceBB):
			#Same-ish BB, load
			faceBB = self.faceBB
		else:
			#New BB, save and skip frame
			self.faceBB = faceBB
			return None

		return faceBB

	#Returns left eye then right (on picture)
	def getEyes(self, face):
		
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

		leftEyeX = 999
		leftEye = None
		rightEye = None

		#Find left and right eyes
		for ex,ey,ew,eh in eyes:
			#New left eye
			if ex < leftEyeX:
				if leftEye is not None:
					rightEye = leftEye
					leftEye = (ex,ey,ew,eh)
				else:
					leftEye = (ex,ey,ew,eh)
					leftEyeX = ex
			else:
				if leftEye is not None:
					rightEye = (ex,ey,ew,eh)

		#Stabilization
		eyesBB = {'left':leftEye,'right':rightEye}

		for side in ['left','right']:

			#Save first frame BB
			if self.eyesBB[side] is None:
				self.eyesBB[side] = eyesBB[side]
			#Load if similar BB
			elif self.isSameEyeBB(eyesBB[side],self.eyesBB[side]):
				eyesBB[side] = self.eyesBB[side]
			#Changed the Bounding Box
			else:
				#New BB, save and skip frame
				self.eyesBB[side] = eyesBB[side] 
				return None
			
		#Get BB for cropping
		xLeft,yLeft,wLeft,hLeft = eyesBB['left']
		xRight,yRight,wRight,hRight = eyesBB['right']

		focusOnCenter = False

		if focusOnCenter:
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

		leftEyeImage = face[yLeft:yLeft+hLeft, xLeft:xLeft+wLeft]
		rightEyeImage = face[yRight:yRight+hRight, xRight:xRight+wRight]

		return leftEyeImage, rightEyeImage










