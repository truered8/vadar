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
TEMP = "temp"
CLASSES = ["car_horn", "dog_bark", "gun_shot", "jackhammer", "siren"]

input_shape = (64, 64, 3)
epochs = 15
batch_size = 32
verbose = 1

# Output: Convolutional Neural Network to train on Mel Spectrograms
def get_model():
	
	model = Sequential()

	# First Convolutional layer
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
	model.add(Activation('relu'))

	# Second Convolutional layer
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# Third Convolutional layer
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))

	# Fourth Convolutional layer
	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	
	# First Dense layer
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	# Output layer
	model.add(Dense(len(CLASSES), activation='softmax'))
	model.compile(keras.optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

	return model

# Predicts one sample
def predict(file, model):
	sample = librosa2mfcc(file)
	sample_reshaped = sample.reshape(1, *input_shape)
	return CLASSES[np.argmax(model.predict(sample_reshaped))]

# Used to copy images into temporary folder
def copy_images(start, destination):
	if not os.path.exists(destination): os.system(f"mkdir {destination}")
	os.system(f"cp -r {start}/* {destination}")


if __name__ == '__main__':
	
	initial = time.time()
	accuracy, val_accuracy, loss, val_loss = 0, 0, 0, 0
	if not os.path.isdir(MODEL_NAME): os.system(f"mkdir {MODEL_NAME}")
	
	# Loading train set and test set
	for i, (train_dirs, valid_dir) in enumerate(get_dirs()):
		print(f'Training on fold {i + 1}.')

		train_gen = ImageDataGenerator(rescale=1./255.)
		valid_gen = ImageDataGenerator(rescale=1./255.)

		# Copy images into main training directory
		for train_dir in train_dirs:
			copy_images(train_dir, TEMP)

		train_generator = train_gen.flow_from_directory(
					        TEMP,
					        target_size=(64, 64),
					        batch_size=batch_size,
					        shuffle=True,
					        class_mode="categorical")

		valid_generator = valid_gen.flow_from_directory(
				        valid_dir,
				        target_size=(64, 64),
				        batch_size=batch_size,
				        shuffle=True,
				        class_mode="categorical")

		STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
		STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

		# Train model
		model = get_model()
		history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs)
		
		# Add results to totals to calculate averages
		accuracy += history.history['accuracy'][-1]
		val_accuracy += history.history['val_accuracy'][-1]
		loss += history.history['loss'][-1]
		val_loss += history.history['val_loss'][-1]
		
		#  Plot accuracy
		plt.style.use('fivethirtyeight')
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(f'{MODEL_NAME}/accuracy-{i+1}.png')
		plt.close()

		# Plot loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(f'{MODEL_NAME}/loss-{i+1}.png')
		plt.close()

		# Clean temporary directory
		os.system(f"rm -r {TEMP}/*")

	# Print average results
	print(f'Average accuracy: {accuracy / 10}')
	print(f'Average validation accuracy: {val_accuracy / 10}')
	print(f'Average loss: {loss / 10}'.format())
	print(f'Average validation loss: {val_loss / 10}')
	print(f'Training took {time.time() - initial} seconds.')
	