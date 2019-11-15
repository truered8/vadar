'''Model Tester

This script is used to make inferences from a trained Convolutional Neural Network on new data. The
data is given as arguments to the program. The Preprocess module is used to generate the MFCCs of the audio data and load them into
memory.

It requires keras to be installed.
'''

from preprocess import wav2mfcc
from keras.models import load_model
import numpy as np
import sys
import time

def predict(model, mfcc):
	reshaped = mfcc.reshape(1, 20, 11, 1)
	return model.predict(reshaped)[0]

PATH_TO_MODEL = 'trained.h5'
LABELS = ['car_horn', 'dog_bark']

initial = time.time()
model = load_model(PATH_TO_MODEL)
print(f'Model took {time.time() - initial} seconds to load.')
for path in sys.argv[1:]:
	initial = time.time()
	prediction = predict(model, wav2mfcc(path))
	print(f'Prediction for {path}: {LABELS[np.argmax(prediction)]}{prediction}; prediction took {time.time() - initial} seconds.')