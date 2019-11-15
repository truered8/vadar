import os
import random
from tqdm import tqdm
fold = 1
i = 1
files = os.listdir('parts')
random.shuffle(files)
for f in tqdm(files):
	if i % (len(files) // 10) == 0:
		fold += 1
	os.system(f'mv parts/{f} UrbanSound8K/audio/fold{fold}/speech')
	i += 1