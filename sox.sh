#!/bin/bash
curdir=`pwd`;
cd "$1";
for i in `find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`; do
	echo $i;
	cd $i;
	for song in *.mp3; do
		echo $song;
		mkdir -p /tmp/$i
		stat 1>/dev/null 2>/dev/null "/tmp/$i/$song.png" || (sox "$song" -n gain -b rate 44100 gain -n remix 1,2 spectrogram -X 50 -y 128 -r -m -o "/tmp/$i/$song.png")
	done
	cd ..;
done
cd $curdir;