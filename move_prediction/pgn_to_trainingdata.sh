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


            bzcat $split_dir/${s}_${c}.pgn.bz2 |

            cat *.pgn > pgns/${s}_${c}.pgn
            rm -v *.pgn


            screen -S "${p_name}-${c}-${s}" -dm bash -c "cd ${s}/${c}; trainingdata-tool -v ../../pgns/${s}_${c}.pgn"
        done
        cd -
    done

done
