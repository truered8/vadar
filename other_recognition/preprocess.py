'''Preprocess

This script is used to preprocess audio data by converting it to MFCCs and save the preprocessed
data to the path specified. 

It requires librosa, numpy, and tqdm to be installed.

It can be imported as a module or run directly as a script.
'''

import os
import librosa
import numpy as np
from tqdm import tqdm

READ_PATH  = '../UrbanSound8K/audio' # Path to read data from
WRITE_PATH = 'preprocessed'          # Path to write preprocessed data

# Input:  Loaded librosa audio file
# Output: MFCC of the audio file
def librosa2mfcc(wave, max_len=11):
	wave = wave[::3]
	wave = librosa.util.normalize(wave)
	mfcc = librosa.feature.mfcc(wave, sr=16000)

	# Pad the MFCC if the maximum length exceeds the MFCC's length
	if (max_len > mfcc.shape[1]):
		pad_width = max_len - mfcc.shape[1]
		mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

	# Else cutoff the remaining parts
	else:
		mfcc = mfcc[:, :max_len]
	
	return mfcc

# Input:  Path to audio file
# Output: MFCC of the audio file
def wav2mfcc(file_path, max_len=11):
	wave, sr = librosa.load(file_path, mono=True, sr=None)
	return librosa2mfcc(wave, max_len)

# Input:  List of labels to preprocess
# Output: Saves preprocessed data to .npy files
def save_data_to_array(labels=['car_horn', 'dog_bark', 'gun_shot', 'siren', 'speech'], path=READ_PATH, max_len=11):

	if not os.path.isdir(WRITE_PATH): os.system(f'mkdir {WRITE_PATH}')
	for fold in range(1, 11):
		print(f'Working on fold {fold}.')
		for label in labels:
			# Initialize MFCC vectors
			mfcc_vectors = []
			wavfiles = [os.path.join(path, f'fold{fold}', label, wavfile) for wavfile in os.listdir(os.path.join(path, f'fold{fold}', label)) if 'wav' in wavfile.lower()]
			for wavfile in tqdm(wavfiles, f'Saving vectors of label - "{label}"'):
				mfcc = wav2mfcc(wavfile, max_len=max_len)
				mfcc_vectors.append(mfcc)
			np.save(f'{WRITE_PATH}/{label}-{fold}.npy', mfcc_vectors)

# Input:  List of classes to load
# Output: Lists of training and validation data and labels to use with 10-fold cross validation
def get_train_test(labels=['car_horn', 'dog_bark', 'gun_shot', 'siren', 'speech']):
	# Initialize lists for all folds
	X_train_all = []
	X_test_all  = []
	y_train_all = []
	y_test_all  = []

	for test_fold in range(1, 11):
		
		# Getting first arrays
		X_test = np.load(f'{WRITE_PATH}/{labels[0]}-{test_fold}.npy')
		y_test = np.zeros(X_test.shape[0])

		# Append all of the dataset into one single array, same goes for y
		for i, label in enumerate(labels[1:]):
			x_test = np.load(f'{WRITE_PATH}/{label}-{test_fold}.npy')
			X_test = np.vstack((X_test, x_test))
			y_test = np.append(y_test, np.full(x_test.shape[0], fill_value=(i + 1)))


		train_folds = list(range(1, 11))
		train_folds.remove(test_fold)

		# Getting first arrays
		X_train = np.load(f'{WRITE_PATH}/{labels[0]}-{train_folds[0]}.npy')
		y_train = np.zeros(X_train.shape[0])
		
		for train_fold in train_folds:
			# Append all of the dataset into one single array, same goes for y
			for i, label in enumerate(labels):
				if train_fold == i == 0: continue
				x_train = np.load(f'{WRITE_PATH}/{label}-{train_fold}.npy')
				X_train = np.vstack((X_train, x_train))
				y_train = np.append(y_train, np.full(x_train.shape[0], fill_value=i))

		X_train_all.append(X_train)
		X_test_all .append(X_test )
		y_train_all.append(y_train)
		y_test_all .append(y_test )

	return zip(X_train_all, X_test_all, y_train_all, y_test_all)

# Input:  List of classes to load
# Output: Training and validation data
def get_all_data(labels=['car_horn', 'dog_bark', 'gun_shot', 'siren', 'speech']):

	# Getting first arrays
	X = np.load(f'{WRITE_PATH}/{labels[0]}-1.npy')
	y = np.zeros(X.shape[0])
	
	for fold in range(1, 11):
		for i, label in enumerate(labels):
			if fold == 1 and i == 0: continue
			x = np.load(f'{WRITE_PATH}/{label}-{fold}.npy')
			X = np.vstack((X, x))
			y = np.append(y, np.full(x.shape[0], fill_value=i))

	return X, y

if __name__ == '__main__':
	save_data_to_array()