#!/bin/bash
cd UrbanSound8K/audio
for fold in $(echo fold*)
do
	printf 'Working on %s.\n'
	cd $fold
	cd speech
	for file in $(echo *.wav)
	do
		audio_degrader -i $file -d gain,-15 mix,sounds/ambience-pub.wav//18 convolution,impulse_responses/ir_smartphone_mic_mono.wav//0.8 dr_compression,2 equalize,50//100//-6 normalization -o "m-$file"
		printf '\tFinished file %s.\n' $file
	done
	cd ../..
done