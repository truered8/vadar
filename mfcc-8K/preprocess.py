import librosa
import os
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import sys

DATA_PATH = "../UrbanSound8K/audio/"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, One-hot encoded labels)
def get_labels(path=DATA_PATH):
	labels = os.listdir(path)
	if '.DS_Store' in labels: labels.remove('.DS_Store')
	label_indices = np.arange(0, len(labels))
	return labels, label_indices, to_categorical(label_indices)

# Input: librosa audio file
# Output: Formatted MFCC of the audio file
def librosa2mfcc(wave, max_len=11):
	wave = wave[::3]
	mfcc = librosa.feature.mfcc(wave, sr=16000)
	
	# If maximum length exceeds mfcc lengths then pad the remaining ones
	if (max_len > mfcc.shape[1]):
		pad_width = max_len - mfcc.shape[1]
		mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

	# Else cutoff the remaining parts
	else:
		mfcc = mfcc[:, :max_len]
	
	return mfcc
	

# Input: Path to audio file
# Output: Formatted MFCC of the audio file
def wav2mfcc(file_path, max_len=11):
	wave, sr = librosa.load(file_path, mono=True, sr=None)
	return librosa2mfcc(wave)
	
# Saves feature vectors of each class to separate files using the above methods
def save_data_to_array(path=DATA_PATH, max_len=11):
	count = 0
	errors = 0
	folds = os.listdir(path)
	folds.remove('generate_data_8K.sh')
	folds.remove('.DS_Store')
	for i, fold in enumerate(folds):
		print('Working on fold {}.'.format(i + 1))
		labels = os.listdir(path + fold)
		labels.remove('.DS_Store')
		for label in labels:
			# Init mfcc vectors
			mfcc_vectors = []

			wavfiles = [path + fold + '/' + label + '/' + wavfile for wavfile in os.listdir(path + fold + '/' + label)]
			for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
				try:
					mfcc = wav2mfcc(wavfile, max_len=max_len)
				except:
					errors += 1
					continue
				count += 1
				mfcc_vectors.append(mfcc)
			np.save('preprocessed/{}-{}.npy'.format(label, i + 1), mfcc_vectors)
			print('Processed {} files with {} errors.'.format(count, errors))


def get_train_test():
	
	X_train_all = []
	y_train_all = []
	X_test_all = []
	y_test_all = []
	labels = ['car_horn', 'dog_bark', 'gun_shot', 'siren']

	for test_fold in range(1, 11):

		# Getting first arrays
		X_test = np.load('preprocessed/{}-{}.npy'.format(labels[0], test_fold))
		y_test = np.zeros(X_test.shape[0])

		# Append all of the dataset into one single array, same goes for y
		for j, label in enumerate(labels[1:]):
			x_test = np.load('preprocessed/{}-{}.npy'.format(label, test_fold))
			X_test = np.vstack((X_test, x_test))
			y_test = np.append(y_test, np.full(x_test.shape[0], fill_value= (j + 1)))


		train_folds = list(range(1, 11))
		train_folds.remove(test_fold)
		for i, train_fold in enumerate(train_folds):
			if i == 0:
				# Getting first arrays
				X_train = np.load('preprocessed/{}-{}.npy'.format(labels[0], train_fold))
				y_train = np.zeros(X_train.shape[0])

			else:
				# Append all of the dataset into one single array, same goes for y
				for j, label in enumerate(labels[1:]):
					x_train = np.load('preprocessed/{}-{}.npy'.format(label, train_fold))
					X_train = np.vstack((X_train, x_train))
					y_train = np.append(y_train, np.full(x_train.shape[0], fill_value= (j + 1)))

		assert X_train.shape[0] == len(y_train)
		assert X_test.shape[0] == len(y_test)

		X_train_all.append(X_train)
		X_test_all.append(X_test)
		y_train_all.append(y_train)
		y_test_all.append(y_test)

	return zip(X_train_all, X_test_all, y_train_all, y_test_all)

def get_all_data():

	labels = ['car_horn', 'dog_bark', 'gun_shot', 'siren']

	# Getting first arrays
	X = np.load('preprocessed/{}-{}.npy'.format(labels[0], 1))
	y = np.zeros(X.shape[0])

	for fold in range(1, 11):
		for i, label in enumerate(labels):
			if fold == 1 and i == 0: continue
			x = np.load('preprocessed/{}-{}.npy'.format(label, fold))
			X = np.vstack((X, x))
			y = np.append(y, np.full(x.shape[0], fill_value=i))
			
	return X, y

if __name__ == '__main__':

	get_all_data()
	'''
	# Save data to array file
	save_data_to_array(max_len=11)
	'''