#!/bin/bash
cd UrbanSound8K/audio
for fold in $(echo fold*)
do
	cd $fold
	mkdir car_horn dog_bark gun_shot siren speech
	mv *-1-*-*.wav car_horn
	mv *-3-*-*.wav dog_bark
	mv *-6-*-*.wav gun_shot
	mv *-8-*-*.wav siren
	cd ..
done