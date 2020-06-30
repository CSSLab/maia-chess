#!/bin/bash
set -e

max_procs=10

#sorry these are relative only
#remove the $PWD to make work with absolute paths
input_file=$PWD/$1
output_files=$PWD/$2
mkdir -p $output_files

mkdir -p $output_files/blocks
mkdir -p $output_files/training
mkdir -p $output_files/validation

cd $output_files/blocks

#using tool from:
#https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/

pgn-extract -7 -C -N  -#1000 $input_file

#use the first 3000 as validation set
mv {1..3}.pgn $output_files/validation/

mv *.pgn $output_files/training/

cd ..
rm -rv $output_files/blocks

for data_type in "training" "validation"; do
    cd $output_files/$data_type
    for p in *.pgn; do
        cd $output_files/$data_type
        p_num=${p%".pgn"}
        echo "Starting on" $data_type $p_num
        mkdir $p_num
        cd $p_num
        #using tool from:
        #https://github.com/DanielUranga/trainingdata-tool
        trainingdata-tool ../$p &
        while [ `echo $(pgrep -c -P$$)` -gt $max_procs ]; do
            printf "waiting\r"
            sleep 1
        done
    done
done
echo "Almost done"
wait
echo "Done"
