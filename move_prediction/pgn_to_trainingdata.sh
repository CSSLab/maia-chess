#!/bin/bash

testing_blocks=4

input_file=$PWD/$1
output_files=$PWD/$2
mkdir -p $output_files

mkdir -p $output_files/blocks
mkdir -p $output_files/training
mkdir -p $output_files/validation

cd $output_files/blocks

pgn-extract -7 -C -N  -#1000 $input_file

#use the first 3000 as validation set
cat {1..3}.pgn > $output_files/validation/validation.pgn
rm {1..3}.pgn

cat *.pgn > $output_files/training/training.pgn

cd $output_files/training
rm -rv $output_files/blocks

trainingdata-tool -v ../../pgns/${s}_${c}.pgn



for player_file in $input_files/*.bz2; do
    f=${player_file##*/}
    p_name=${f%.pgn.bz2}
    p_dir=$output_files/$p_name
    split_dir=$output_files/$p_name/split
    mkdir -p $p_dir
    mkdir -p $split_dir
    echo $p_name $p_dir
    python split_by_player.py $player_file $p_name $split_dir/games


    for c in "white" "black"; do
        python pgn_fractional_split.py $split_dir/games_$c.pgn.bz2 $split_dir/train_$c.pgn.bz2 $split_dir/validate_$c.pgn.bz2 $split_dir/test_$c.pgn.bz2 --ratios $train_frac $val_frac $test_frac

        cd $p_dir
        mkdir -p pgns
        for s in "train" "validate" "test"; do
            mkdir -p $s
            mkdir $s/$c

            #using tool from:
            #https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/
            bzcat $split_dir/${s}_${c}.pgn.bz2 |

            cat *.pgn > pgns/${s}_${c}.pgn
            rm -v *.pgn

             #using tool from:
            #https://github.com/DanielUranga/trainingdata-tool
            screen -S "${p_name}-${c}-${s}" -dm bash -c "cd ${s}/${c}; trainingdata-tool -v ../../pgns/${s}_${c}.pgn"
        done
        cd -
    done

done
