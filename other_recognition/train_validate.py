'''Train and Validate

This script is used to train a Convolutional Neural Network to recognize audio data using 10-fold
cross validation. The Preprocess module is used to generate the MFCCs of the audio data and load
them into memory.

It requires librosa, sklearn, and keras to be installed.
'''

from preprocess import *
import keras
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import regularizers
from sklearn.utils import class_weight

# Path to save graphs and results
PATH = 'trained_8'
LABELS = ['car_horn', 'dog_bark']

# Input data dimensions
feature_dim_1 = 20
feature_dim_2 = 11
channel = 1


# Input:  None
# Output: Convolutional Neural Network to train on MFCCs
def get_model():

	# Hyperparameters
	n_filters_1 = 16
	n_filters_2 = 32
	d_filter = 3
	p_drop_1 = 0.50
	p_drop_2 = 0.60
	reg = 0.004

	epochs = 85
	batch_size = 256
	verbose = 1

	model = Sequential()

	model.add(Conv2D(n_filters_1, kernel_size=(d_filter, d_filter), activation='relu', kernel_regularizer=regularizers.l2(reg), input_shape=(feature_dim_1, feature_dim_2, channel)))
	model.add(BatchNormalization())

	# Second layer
	model.add(Conv2D(n_filters_1, kernel_size=(d_filter, d_filter), activation='relu', kernel_regularizer=regularizers.l2(reg)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Drop layer
	model.add(Dropout(p_drop_1))

	## Used to flat the input (1, 10, 2, 2) -> (1, 40)
	model.add(Flatten())

	# Full Connected layer
	model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg)))
	# Drop layer
	model.add(Dropout(p_drop_2))
	# Full Connected layer
	model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(reg)))
	# Drop layer
	model.add(Dropout(p_drop_2))
	# Output Full Connected layer
	model.add(Dense(len(LABELS), activation='softmax'))
	model.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['accuracy'])

	return model, epochs, batch_size, verbose

# Input:  None
# Output: Original Convolutional Neural Network used to train on MFCCs
def get_original_model():

	# Hyperparameters
	n_filters_1 = 32
	n_filters_2 = 64
	d_filter = 2

	epochs = 75
	batch_size = 72
	verbose = 1

	model = Sequential()

	model.add(Conv2D(n_filters_1, kernel_size=(d_filter, d_filter), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))

	# Second layer
	model.add(Conv2D(n_filters_1, kernel_size=(d_filter, d_filter), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	## Used to flat the input (1, 10, 2, 2) -> (1, 40)
	model.add(Flatten())

	# Full Connected layer
	model.add(Dense(64, activation='relu'))
	# Full Connected layer
	model.add(Dense(32, activation='relu'))
	# Output Full Connected layer
	model.add(Dense(len(LABELS), activation='softmax'))
	model.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['accuracy'])

	return model, epochs, batch_size, verbose

# Input:  Loaded librosa audio file, loaded Convolutional Neural Network
# Output: Prediction of the Convolutional Neural Network
def predict(file, model):
	sample = librosa2mfcc(file)
	sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
	return LABELS[np.argmax(model.predict(sample_reshaped))]


if __name__ == '__main__':
	
	if not os.path.isdir(PATH): os.system(f'mkdir {PATH}')
	initial = time.time()
	acc, val_acc, loss, val_loss = 0, 0, 0, 0

	# Train using 10-fold cross validation
	for i, (X_train, X_test, y_train, y_test) in enumerate(get_train_test(labels=LABELS)):
		print(f'Training on fold {i + 1}.')

		# Reshaping to perform 2D convolution
		X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
		X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
		
		y_train_hot = to_categorical(y_train)
		y_test_hot = to_categorical(y_test)

		# Class weights to counter imbalanced classes
		class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)))
		
		# Train model
		model, epochs, batch_size, verbose = get_model()
		history = model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, class_weight=class_weights, validation_data=(X_test, y_test_hot))

		# Add results to totals to calculate averages
		acc += history.history['acc'][-1]
		val_acc += history.history['val_acc'][-1]
		loss += history.history['loss'][-1]
		val_loss += history.history['val_loss'][-1]
		
		#  Plot accuracy
		plt.style.use('fivethirtyeight')
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(f'{PATH}/acc-{i}.png')
		plt.close()

		# Plot loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(f'{PATH}/loss-{i}.png')
		plt.close()

	# Print and save average results
	print(f'Average accuracy: {acc / 10}')
	print(f'Average validation accuracy: {val_acc / 10}')
	print(f'Average loss: {loss / 10}'.format())
	print(f'Average validation loss: {val_loss / 10}')
	print(f'Training took {time.time() - initial} seconds.')

	results = open(f'{PATH}/results.txt', 'w')
	results.write(f'Average accuracy: {acc / 10}\n')
	results.write(f'Average validation accuracy: {val_acc / 10}\n')
	results.write(f'Average loss: {loss / 10}\n')
	results.write(f'Average validation loss: {val_loss / 10}\n')
	results.write(f'Training took {time.time() - initial} seconds.')
	results.close()