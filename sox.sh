#!/bin/bash
for i in `find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`; do
	echo $i;
	cd $i;
	for song in *.mp3; do
		len=`sox "$song" -n stat 2>&1 | sed -n 's#^Length (seconds):[^0-9]*\([0-9.]*\)$#\1#p'`
		len=`calc $len/2`
		begin=`calc $len-1.28 | xargs`
		end=`calc $len+1.28 | xargs`
		echo "$song `calc $len*2|xargs` ($begin s, $end s)";
		stat 1>/dev/null 2>/dev/null "$song.png" || (sox "$song" -n trim $begin 2.56 gain -b rate 44100 gain -n remix 1,2 spectrogram -X 50 -y 128 -r -m -o "$song.png")
	done
	cd ..;
done
cd $curdir;
