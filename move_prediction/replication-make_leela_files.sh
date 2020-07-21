#!/bin/bash


mkdir ../data/elo_ranges
cw=`pwd`
for elo in {1100..1900..100}; do
    echo $i
    mkdir "../data/elo_ranges/${elo}"
    outputtest="../data/elo_ranges/${elo}/test"
    outputtrain="../data/elo_ranges/${elo}/train"
    mkdir $outputtest
    mkdir $outputtrain
    for te in "../data/final_training_data/pgns_ranged_training/${elo}"/*; do
        fname="$(basename -- $te)"
        echo "${elo}-${fname}"
        cd $outputtrain
        mkdir $fname
        cd $fname
        screen -S "${elo}-${fname}-test" -dm bash -c "trainingdata-tool -v -files-per-dir 5000 ${te}"
        cd ..
    done
    for te in "../data/final_training_data/pgns_ranged_testing/${elo}"/{1..2}.pgn; do
        fname="$(basename -- $te)"
        echo "${elo}-${fname}"
        cd $outputtest
        mkdir $fname
        cd $fname
        echo "trainingdata-tool -v -files-per-dir 5000 ${te}"
        screen -S "${elo}-${fname}-test" -dm bash -c "trainingdata-tool -v -files-per-dir 5000 ${te}"
        cd ..
    done
    te="../data/final_training_data/pgns_ranged_testing/${elo}/3.pgn"    fname="$(basename -- $te)"
    echo "${elo}-${fname}"
    cd $outputtest
    mkdir $fname
    cd $fname
    trainingdata-tool -v -files-per-dir 5000 ${te}
    cd ..
done
cd $cw






#After merging split pgns
pgn-extract -7 -C -N  -#400000 /datadrive/pgns_ranged/1200/lichess_1200.pgn
pgn-extract -7 -C -N  -#400000 /datadrive/pgns_ranged/1500/lichess_1500.pgn
pgn-extract -7 -C -N  -#400000 /datadrive/pgns_ranged/1800/lichess_1800.pgn


#Then on all the results

trainingdata-tool -v -files-per-dir 5000 lichess_1800.pgn
for f in *.pgn; do echo "${f%.*}"; mkdir "${f%.*}_files"; cd "${f%.*}_files"; trainingdata-tool -v -files-per-dir 5000 "../${f}"; cd ..; done

for f in {1..10}.pgn; do echo "${f%.*}"; mkdir "${f%.*}_files"; cd "${f%.*}_files"; trainingdata-tool -v -files-per-dir 5000 "../${f}"; cd ..; done

mkdir train
mkdir test
mv 10_files/ test/
mv 10.pgn test/
mv *_* train/
mv *.pgn  train


download pgns_ranged.zip
unzip pgns_ranged.zip
cd pgns_ranged
for elo in *; do
    echo $elo
    cd $elo
    for year in *.pgn.bz2; do
        echo "${year%.*.*}"
        mkdir "${year%.*.*}"
        cd "${year%.*.*}"
        screen -S "${elo}-${year%.*.*}" -dm bash -c "bzcat \"../${year}\" | pgn-extract -7 -C -N  -#400000"
        cd ..
    done
    cd ..
done

for elo in *; do
    echo $elo
    cd $elo
    mkdir -p train
    mkdir -p test
    for year in lichess-*/; do
        yearonly="${year#lichess-}"
        yearonly="${yearonly%/}"
        echo "${elo}-${yearonly}"
        cd test
        mkdir -p "${yearonly}"
        mkdir -p "${yearonly}/1"
        cd "${yearonly}/1"

        screen -S "${elo}-${yearonly}-test" -dm bash -c "trainingdata-tool -v -files-per-dir 5000 \"../../../${year}/1.pgn\""

        cd ../../..
        cd train
        mkdir -p "${yearonly}"
        cd "${yearonly}"
        for i in {2..10}; do
            echo "${i}"
            mkdir -p "${i}"
            cd "${i}"
            screen -S "${elo}-${yearonly}-train-${i}" -dm bash -c "trainingdata-tool -v -files-per-dir 5000 \"../../../${year}/${i}.pgn\""
            cd ..
        done
        cd ../..
    done
    cd ..
done

for scr in $(screen -ls | awk '{print $1}'); do if [[ $scr == *"test"* ]]; then echo $scr; screen -S $scr -X kill; fi; done

for scr in $(screen -ls | awk '{print $1}'); do if [[ $scr == *"2200"* ]]; then echo $scr; screen -S $scr -X kill; fi; done


for scr in $(screen -ls | awk '{print $1}'); do if [[ $scr == *"final"* ]]; then echo $scr; screen -S $scr -X kill; fi; done
