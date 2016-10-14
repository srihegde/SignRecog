!#bin/sh

############################################################# 
# A bash script for converting .mpg files to .avi in batch
############################################################# 

for file in "./resources/videos"/*
do
	ffmpeg -i "$file" ${file%%.mpg}.avi
	# echo ${file%%.mpg}.avi
done