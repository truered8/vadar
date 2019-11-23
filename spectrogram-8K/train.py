from preprocess import *
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import regularizers
import time

# Second dimension of the feature is dim2
feature_dim_2 = 11

# # Feature dimension
feature_dim_1 = 20

# Hyperparameters
n_filters_1 = 32
n_filters_2 = 64
d_filter = 2
p_drop_1 = 0.25
p_drop_2 = 0.50
reg = 0.002

channel = 1
epochs = 100
batch_size = 72
verbose = 1
num_classes = 4

# Output: Convolutional Neural Network to train on MFCCs
def get_model():
	'''
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
	model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
	model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adadelta(),
				  metrics=['accuracy'])
	return model
	'''
	
	model = Sequential()
	## NET MODEL 0:
	#
	# INPUT -> [CONV -> RELU -> CONV -> RELU -> POLL] ->
	# -> [CONV -> RELU -> CONV -> RELU -> POLL] -> FC -> RELU -> FC
	#
	# - IMPLEMENTED METHOD-

	# First layer
	model.add(Conv2D(n_filters_1, kernel_size=(d_filter, d_filter), activation='relu', kernel_regularizer=regularizers.l2(reg), input_shape=(feature_dim_1, feature_dim_2, channel)))
	model.add(BatchNormalization())

	# Second layer
	model.add(Conv2D(n_filters_1, kernel_size=(d_filter, d_filter), activation='relu', kernel_regularizer=regularizers.l2(reg)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Drop layer
	model.add(Dropout(p_drop_1))
	
	# Third layer
	model.add(Conv2D(n_filters_2, kernel_size=(d_filter, d_filter), activation='relu', kernel_regularizer=regularizers.l2(reg)))
	model.add(BatchNormalization())

	# Fouth layer
	model.add(Conv2D(n_filters_2, kernel_size=(d_filter, d_filter), activation='relu', kernel_regularizer=regularizers.l2(reg)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Drop layer
	model.add(Dropout(p_drop_1))

	## Used to flat the input (1, 10, 2, 2) -> (1, 40)
	model.add(Flatten())

	# Full Connected layer
	model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(reg)))
	# Drop layer
	model.add(Dropout(p_drop_2))
	# Full Connected layer
	model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(reg)))
	# Drop layer
	model.add(Dropout(p_drop_2))
	# Output Full Connected layer
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
			  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
			  metrics=['accuracy'])

	return model

def light_model():
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
	model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(reg)))
	# Drop layer
	model.add(Dropout(p_drop_2))
	# Full Connected layer
	model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(reg)))
	# Drop layer
	model.add(Dropout(p_drop_2))
	# Output Full Connected layer
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
			  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
			  metrics=['accuracy'])

	return model

def test_model():
	model = Sequential()
	model.add(Conv2D(8, kernel_size=(2, 2), activation='relu', kernel_regularizer=regularizers.l2(reg), input_shape=(feature_dim_1, feature_dim_2, channel)))
	model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
			  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
			  metrics=['accuracy'])
	return model

if __name__ == '__main__':

	X, y = get_all_data()
	
	# Reshaping to perform 2D convolution
	X = X.reshape(X.shape[0], feature_dim_1, feature_dim_2, channel)
	y_hot = to_categorical(y)
	
	# Train model
	model = light_model()
	initial = time.time()
	history = model.fit(X, y_hot, batch_size=batch_size, epochs=epochs, verbose=verbose)
	print('Training took {} seconds.'.format(time.time() - initial))
	model.save('model.h5')

	#  Plot accuracy
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
