#!/bin/bash
# gpu or cpu
postfix=$1
# folder to input data without gpu or cpu postfix
input_path=out_
# folder to output data without gpu or cpu postfix
out_path=images_
# create output folder
mkdir "$out_path""$postfix"
# clear folder
rm "$out_path""$postfix"/*
# convert cycle
for file_name in ${input_path}${postfix}/*.data; do
    without_ext=${file_name%.data}
    image_name=${without_ext#*/}
    python converter.py "$file_name" "$out_path""$postfix"/"$image_name".png
done