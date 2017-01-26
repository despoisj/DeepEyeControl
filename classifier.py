import threading
import time
from config import datasetImageSize, datasetMaxSerieLength, modelsPath, framesInHistory
from model import createModel
from datasetTools import padLSTM

#What we want to reach
predictionsPerSecond = 2

#Approximate duration of a motion (s)
motionDuration = 1.

#/!\ HARD-CODED estimation
recordingFPS = 25.

#Time to replace all history (s)  84/25 = 2.56s
historyDuration = framesInHistory/recordingFPS

#How many predictions to have average on motion
averageWindowSize = int((historyDuration-motionDuration)*predictionsPerSecond+1)

try:
	from terminaltables import AsciiTable
	table_data = [
		['Parameter', 'Value'],
		['Predictions/s (target)', predictionsPerSecond],
		['Frames recorded/s (estimate)', recordingFPS],
		['Time to teplace all history (s)', historyDuration],
		['Motion lifespan in history (s)', (historyDuration-motionDuration)],
		['Prediction needed for average', averageWindowSize]
	]
	print AsciiTable(table_data).table
except ImportError:
	print "======== Classification parameters ========"
	print predictionsPerSecond, "predictions/s (classifier target)"
	print recordingFPS, "frames/s (recording target)"
	print historyDuration, "s to replace all history"
	print (historyDuration-motionDuration), "s (motion lifespan in history)" 
	print averageWindowSize, "predictions needed to average"
	print "===========================================\n"
	

def threaded(fn):
	def wrapper(*args, **kwargs):
		thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
		thread.start()
		return thread
	return wrapper

class Classifier:	
	def __init__(self):
		self.X0 = None
		self.predictions = []
		self.lastPredictions = None
		self.model = createModel(nbClasses=3, imageSize=datasetImageSize, maxlength=datasetMaxSerieLength)
		print "Loading model parameters..."
		self.model.load(modelsPath+'eyeDNN_HD_SLIDE.tflearn')
		
		#History of last 3 predictions
		self.history = [None for i in range(averageWindowSize)]

	@threaded
	def startPredictions(self):
		while True:
			if self.X0 is not None:

				#Record start time
				t0 = time.time()

				#Read history
				X0 = list(self.X0)
				X1 = list(self.X1)

				#Pad inputs
				X0 = padLSTM(X0, maxlen=datasetMaxSerieLength, padding='post', value=0.)
				X1 = padLSTM(X0, maxlen=datasetMaxSerieLength, padding='post', value=0.)

				#Get predictions from the model with post padding			
				predictionSoftmax = self.model.predict([X0,X1])[0]
				predictedIndex = max(enumerate(predictionSoftmax), key=lambda x:x[1])[0]
				print "Prediction:", ["{0:.2f}".format(x) for x in predictionSoftmax], "->", predictedIndex
				
				self.predictions.append((predictionSoftmax,predictedIndex))
				self.lastPredictions = (predictionSoftmax,predictedIndex)

				#Crop beginning
				self.history = self.history[1:]
				self.history.append(predictionSoftmax)

				if None not in self.history:
					averageSoftmax = [sum(_)/float(len(self.history)) for _ in zip(*self.history)]
					averageIndex = max(enumerate(averageSoftmax), key=lambda x:x[1])[0]
					print "AVERAGE ========================-:", ["{0:.2f}".format(x) for x in averageSoftmax], "->", averageIndex

				#FPS ?
				#Tiny sleep ?
				timeElapsed = time.time() - t0
				#Try to maintain 1./predictionsPerSecond s between each prediction
				time.sleep(max(0, 1./predictionsPerSecond - timeElapsed))





