#!/bin/bash
for i in `find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`; do
	cd $i;
	for song in *.mp3; do
		echo $song;
#		sox "$song" -n rate 44100 remix 1,2 spectrogram -X 50 -y 128 -r -m -o "$song.png"
	done
	cd ..;
done