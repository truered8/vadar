print('Importing libraries...')
import librosa
from keras.models import load_model
import numpy as np
import sys

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

def predict(file, model):
    sample = librosa2mfcc(file)
    sample_reshaped = sample.reshape(1, 20, 11, 1)
    predictions = model.predict(sample_reshaped)
    return LABELS[np.argmax(predictions)]

LABELS = ['Car Horn', 'Dog Bark', 'Gun Shot', 'Siren']
model = load_model('trained.h5')
print('Predicting...')
for path in sys.argv[1:]:
	data, _ = librosa.load(path, mono=True, sr=None)
	print(predict(data, model))
