print("Importing libraries...")
import librosa
import librosa.display
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

DATA_PATH = "../UrbanSound8K/audio"
SPEECH_PATH = "../wav"
SAVE_PATH = "preprocessed"
ID_TO_CLASS = {0: "air_conditioner",
			   1: "car_horn",
			   2: "children_playing",
			   3: "dog_bark",
			   4: "drilling",
			   5: "engine_idling",
			   6: "gun_shot",
			   7: "jackhammer",
			   8: "siren",
			   9: "street_music"}
CLASSES = ["car_horn", "dog_bark", "gun_shot", "jackhammer", "siren"]

# https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4
def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = SAVE_PATH + "/" + name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def save_data():
	if not os.path.isdir(SAVE_PATH): os.system(f"mkdir {SAVE_PATH}")
	for i in range(1, 11): 
		if not os.path.isdir(f"{SAVE_PATH}/fold{i}"): os.system(f"mkdir {SAVE_PATH}fold{i}")
		for label in CLASSES:
			if not os.path.isdir(f"{SAVE_PATH}/fold{i}/{label}"): os.system(f"mkdir {SAVE_PATH}fold{i}/{label}")
	for fold in os.listdir(DATA_PATH):
		if fold == ".DS_Store": continue
		i = 0
		for clip in tqdm(os.listdir(DATA_PATH + "/" + fold), desc=f"Working on fold {fold[4:]}"):
			if clip == ".DS_Store": continue
			label = ID_TO_CLASS[int(clip.split("-")[1])]
			if label not in CLASSES: continue
			create_spectrogram(DATA_PATH + "/" + fold + "/" + clip, f"fold{int(fold[4:])}/{label}/{label}-{i}")
			i += 1
	print("Finished saving data.")

def save_speech_data():
	for i in range(1, 11): 
		if not os.path.isdir(f"{SAVE_PATH}/fold{i}/speech"): os.system(f"mkdir {SAVE_PATH}/fold{i}/speech")
	i = 0
	for idnum in os.listdir(SPEECH_PATH):
		if idnum == ".DS_Store": continue
		for directory in tqdm(os.listdir(f"{SPEECH_PATH}/{idnum}")[:20]):
			if directory == ".DS_Store": continue
			for clip in os.listdir(f"{SPEECH_PATH}/{idnum}/{directory}"):
				fold = f"fold{i%10+1}"
				create_spectrogram(f"{SPEECH_PATH}/{idnum}/{directory}/{clip}", f"fold{i%10+1}/speech/speech-{i}")
				i += 1
	print("Finished saving speech data.")

def delete():
	for fold in os.listdir(SAVE_PATH):
		if fold == ".DS_Store": continue
		i = 0
		for img in tqdm(os.listdir(f"{SAVE_PATH}/{fold}/speech"), desc=f"Working on fold {fold[4:]}"):
			if not i % 10 == 0:
				os.system(f"rm {SAVE_PATH}/{fold}/speech/{img}")
			i += 1

def get_dirs():

	train_all = []
	valid_all = []

	for valid_fold in os.listdir(SAVE_PATH):
		if not valid_fold[:4] == "fold": continue

		train_folds = [f"{SAVE_PATH}fold{i}" for i in range(1, 11)]
		train_folds.remove(f"{SAVE_PATH}{valid_fold}")

		train_all.append(train_folds)
		valid_all.append(f"{SAVE_PATH}{valid_fold}")

	return zip(train_all, valid_all)


if __name__ == '__main__':
	save_data()
	save_speech_data()