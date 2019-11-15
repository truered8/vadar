import librosa
import cv2
import numpy as np
import os
from tqdm import tqdm

MAX_SHIFT = 2400

for fold in [f for f in os.listdir(os.path.join('UrbanSound8K', 'audio')) if 'fold' in f]:
	print(f'Working on {fold}.')
	for label in [l for l in os.listdir(os.path.join('UrbanSound8K', 'audio', fold)) if os.path.isdir(os.path.join('UrbanSound8K', 'audio', fold, l))]:
		for file in tqdm([os.path.join('UrbanSound8K', 'audio', fold, label, f) for f in os.listdir(os.path.join('UrbanSound8K', 'audio', fold, label)) if 'wav' in f], f'Working on label {label}'):
			wav, sr = librosa.load(file, mono=True, sr=None)

			# Time shifting
			start_ = int(np.random.uniform(-MAX_SHIFT,MAX_SHIFT))
			if start_ >= 0:
				wav = np.r_[wav[start_:], np.random.uniform(-0.001,0.001, start_)]
			else:
				wav = np.r_[np.random.uniform(-0.001,0.001, -start_), wav[:start_]]

			# Speed Tuning
			speed_rate = np.random.uniform(0.7,1.3)
			wav = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
			if len(wav) < 16000:
				pad_len = 16000 - len(wav)
				wav = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
									   wav,
									   np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
			else: 
				cut_len = len(wav) - 16000
				wav = wav[int(cut_len/2):int(cut_len/2)+16000]

			# Volume Tuning
			wav = wav * np.random.uniform(0.8, 1.2)
			librosa.output.write_wav(file[:-4] + '-a.wav', wav, sr)