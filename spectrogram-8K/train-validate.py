from preprocess import *
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import time

MODEL_NAME = "alpha"
DATA_PATH = "preprocessed/"

# Second dimension of the feature is dim2
feature_dim_2 = 11

# # Feature dimension
feature_dim_1 = 20

# Hyperparameters
n_filters_1 = 32
n_filters_2 = 64
d_filter = 2
p_drop_1 = 0.40
p_drop_2 = 0.50
reg = 0.0025

channel = 1
epochs = 150
batch_size = 72
verbose = 1
num_classes = 4

# Output: Convolutional Neural Network to train on MFCCs
def get_model():
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
	model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(reg)))
	# Drop layer
	model.add(Dropout(p_drop_2))
	# Full Connected layer
	model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1(reg)))
	# Drop layer
	model.add(Dropout(p_drop_2))
	# Output Full Connected layer
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
			  optimizer=SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True),
			  metrics=['accuracy'])

	return model

# Predicts one sample
def predict(file, model):
	sample = librosa2mfcc(file)
	sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
	return get_labels()[0][
			np.argmax(model.predict(sample_reshaped))
	]


if __name__ == '__main__':
	
	initial = time.time()
	acc, val_acc, loss, val_loss = 0, 0, 0, 0

	# Loading train set and test set
	for i, (X_train, X_test, y_train, y_test) in enumerate(get_train_test()):
		print('Training on fold {}.'.format(i))

		train_gen = ImageDataGenerator(rescale=1./255.)
		valid_gen = ImageDataGenerator(rescale=1./255.)

		train_generator = train_gen.flow_from_directory(
				        train_dir,
				        target_size=(150, 150),
				        batch_size=20,
				        class_mode='binary')
		valid_generator = valid_gen.flow_from_directory(
				        valid_dir,
				        target_size=(150, 150),
				        batch_size=20,
				        class_mode='binary')

		# Train model
		model = get_model()
		history = model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))

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
		plt.savefig(f'{MODEL_NAME}/acc-{i}.png')
		plt.close()

		# Plot loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(f'{MODEL_NAME}/loss-{i}.png')
		plt.close()

	# Print average results
	print(f'Average accuracy: {acc / 10}')
	print(f'Average validation accuracy: {val_acc / 10}')
	print(f'Average loss: {loss / 10}'.format())
	print(f'Average validation loss: {val_loss / 10}')
	print(f'Training took {time.time() - initial} seconds.')