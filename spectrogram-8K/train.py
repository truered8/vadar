from preprocess import *
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import regularizers
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

	# Copy images into main training directory
	for train_dir in os.listdir(DATA_PATH):
		copy_images(f"{DATA_PATH}{train_dir}", TEMP)
	
	train_generator = train_gen.flow_from_directory(
					        TEMP,
					        target_size=(64, 64),
					        batch_size=batch_size,
					        shuffle=True,
					        class_mode="categorical")
	STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
	
	# Train model
	model = get_model()
	initial = time.time()
	history = history = model.fit_generator(generator=train_generator,
						                    steps_per_epoch=STEP_SIZE_TRAIN,
						                    epochs=epochs)
	print(f"Training took {time.time() - initial} seconds.")
	model.save(f'{MODEL_NAME}/{MODEL_NAME}.h5')

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

	# Clean temporary directory
	os.system(f"rm -r {TEMP}/*")