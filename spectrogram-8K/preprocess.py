print("Importing libraries...")
import librosa
import librosa.display
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

DATA_PATH = "../UrbanSound8K/audio/"
SAVE_PATH = "preprocessed/"
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
    filename  = SAVE_PATH + name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def save_data():
	if not os.path.isdir(SAVE_PATH): os.system(f"mkdir {SAVE_PATH}")
	for i in range(1, 11): 
		if not os.path.isdir("{SAVE_PATH}fold{i}"): os.system(f"mkdir {SAVE_PATH}fold{i}")
	for fold in os.listdir(DATA_PATH):
		if fold == ".DS_Store": continue
		i = 0
		for clip in tqdm(os.listdir(DATA_PATH + "/" + fold), desc=f"Working on fold {fold[4:]}"):
			if clip == ".DS_Store": continue
			label = ID_TO_CLASS[int(clip.split("-")[1])]
			if label not in CLASSES: continue
			create_spectrogram(DATA_PATH + "/" + fold + "/" + clip, f"fold{int(fold[4:])}/{label}-{i}")
			i += 1
	print("Finished saving data.")

def get_train_test(labels):

	train_all = []
	valid_all = []

	for valid_fold in os.listdir(preprocessed):
		if valid_fold == ".DS_Store": continue

		train_folds = [f"fold{i}" for i in range(1, 11)]
		train_folds.remove(test_fold)

		train_all.append(train_folds)
		valid_all.append(valid_fold)

	return zip(train_all, valid_all)


if __name__ == '__main__':
	save_data()