#!/bin/bash

#Read the raw pgns from lichess and filter out the elo ranges we care about to make our validation set

mkdir ../data/pgns_traj_testing/
for i in {1000..2500..100}; do
    echo $i
    upperval=$(($i + 100))
    screen -S "${i}-testing" -dm bash -c "source ~/.bashrc; python3 replication-extractELOrange.py --remove_bullet ${i} ${upperval} ../data/pgns_traj_testing/${i}_2019-12.pgn.bz2 ../datasets/lichess_db_standard_rated_2019-12.pgn.bz2;bzcat ../data/pgns_traj_testing/${i}_2019-12.pgn.bz2 | pgn-extract -Wuci | uci-analysis --engine stockfish --searchdepth 15 --bookdepth 0 --annotatePGN | pgn-extract --output ../data/pgns_traj_testing/${i}_2019-12_anotated.pgn"
done

#Don't really need screen for this
mkdir ../data/pgns_traj_blocks/
for i in {1000..2500..100}; do
    echo $i
    screen -S "${i}-testing-split" -dm bash -c "bzcat ../data/pgns_traj_testing/${i}_2019-12.pgn.bz2 | pgn-extract --stopafter 10000 --output ../data/pgns_traj_blocks/${i}_10000_2019-12.pgn"
done

mkdir ../data/pgns_traj_csvs/
for i in {1000..2500..100}; do
    echo $i
    screen -S "${i}-testing-csv" -dm bash -c "source ~/.bashrc; python3 ../data_generators/make_month_csv.py --allow_non_sf ../data/pgns_traj_blocks/${i}_10000_2019-12.pgn ../data/pgns_traj_csvs/"
done
