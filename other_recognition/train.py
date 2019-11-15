'''Train and Save Model

This script is used to train a Convolutional Neural Network to recognize audio data and save the
model. The Preprocess module is used to generate the MFCCs of the audio data and load them into
memory.

It requires librosa, sklearn, and keras to be installed.
'''

from preprocess import *
from train_validate import *
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import regularizers
from sklearn.utils import class_weight
import time

# Path to save model
SAVE_PATH = 'trained.h5'

if __name__ == '__main__':
	
	initial = time.time()

	# Load all data
	X, y = get_all_data(LABELS)

	# Class weights to counter imbalanced classes
	class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y), y)))

	# Reshaping to perform 2D convolution
	X = X.reshape(X.shape[0], feature_dim_1, feature_dim_2, channel)
	y = to_categorical(y)

	
	# Train model
	model, epochs, batch_size, verbose = get_model()
	history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose, class_weight=class_weights)
	model.save(SAVE_PATH)

	#  Plot accuracy
	plt.style.use('fivethirtyeight')
	plt.plot(history.history['acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.show()

	# Plot loss
	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.show()