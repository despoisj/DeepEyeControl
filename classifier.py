import threading
import time
from config import dataset_img_size, dataset_max_serie_length, models_path, frames_in_history
from model import create_model
from dataset_tools import pad_lstm

#What we want to reach
predictions_per_second = 2

#Approximate duration of a motion (s)
motion_duration = 1.

#/!\ HARD-CODED estimation
recording_fps = 25.

#Time to replace all history (s)  84/25 = 2.56s
history_duration = frames_in_history / recording_fps

#How many predictions to have average on motion
average_window_size = int((history_duration - motion_duration) * predictions_per_second + 1)

try:
	from terminaltables import AsciiTable
	table_data = [
		['Parameter', 'Value'],
		['Predictions/s (target)', predictions_per_second],
		['Frames recorded/s (estimate)', recording_fps],
		['Time to teplace all history (s)', history_duration],
		['Motion lifespan in history (s)', (history_duration - motion_duration)],
		['Prediction needed for average', average_window_size]
	]
	print AsciiTable(table_data).table
except ImportError:
	print("======== Classification parameters ========")
	print(predictions_per_second, "predictions/s (classifier target)")
	print(recording_fps, "frames/s (recording target)")
	print(history_duration, "s to replace all history")
	print((history_duration-motion_duration), "s (motion lifespan in history)")
	print(average_window_size, "predictions needed to average")
	print("===========================================\n")
	

def threaded(fn):
	def wrapper(*args, **kwargs):
		thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
		thread.start()
		return thread
	return wrapper


class Classifier:
	"""Threaded classifier to make prediction as fast as possible based on history of predictions"""
	def __init__(self):
		self.X0 = None
		self.predictions = []
		self.last_predictions = None
		self.model = create_model(nb_classes=3, img_size=dataset_img_size, max_length=dataset_max_serie_length)
		print("Loading model parameters...")
		self.model.load(models_path + 'eyeDNN_HD_SLIDE.tflearn')
		
		#History of last 3 predictions
		self.history = [None for i in range(average_window_size)]

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
				X0 = pad_lstm(X0, maxlen=dataset_max_serie_length, padding='post', value=0.)
				X1 = pad_lstm(X0, maxlen=dataset_max_serie_length, padding='post', value=0.)

				#Get predictions from the model with post padding			
				prediction_softmax = self.model.predict([X0,X1])[0]
				predicted_index = max(enumerate(prediction_softmax), key=lambda x:x[1])[0]
				print("Prediction:", ["{0:.2f}".format(x) for x in prediction_softmax], "->", predicted_index)
				
				self.predictions.append((prediction_softmax, predicted_index))
				self.last_predictions = (prediction_softmax, predicted_index)

				#Crop beginning
				self.history = self.history[1:]
				self.history.append(prediction_softmax)

				if None not in self.history:
					average_softmax = [sum(_)/float(len(self.history)) for _ in zip(*self.history)]
					average_index = max(enumerate(average_softmax), key=lambda x:x[1])[0]
					print("AVERAGE ========================-:", ["{0:.2f}".format(x) for x in average_softmax], "->", average_index)

				#TODO FPS ?
				#TODO Tiny sleep ?
				timeElapsed = time.time() - t0
				#Try to maintain 1./predictions_per_second s between each prediction
				time.sleep(max(0, 1. / predictions_per_second - timeElapsed))
