import pandas
import bz2
import argparse
import os

import haibrid_chess_utils

target_columns = ['game_id', 'type', 'time_control', 'num_ply', 'move_ply', 'move', 'cp', 'cp_rel', 'cp_loss', 'is_blunder', 'winrate', 'winrate_loss', 'blunder_wr', 'is_capture', 'opp_winrate', 'white_active', 'active_elo', 'opponent_elo', 'active_won', 'clock', 'opp_clock', 'board']

@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='Create new cvs with select columns', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input', help='input CSV')
    parser.add_argument('outputDir', help='output CSV')

    args = parser.parse_args()

    haibrid_chess_utils.printWithDate(f"Starting CSV conversion of {args.input} writing to {args.outputDir}")
    haibrid_chess_utils.printWithDate(f"Collecting {', '.join(target_columns)}")

    name = os.path.basename(args.input).split('.')[0]
    outputName = os.path.join(args.outputDir, f"{name}_trimmed.csv.bz2")

    haibrid_chess_utils.printWithDate(f"Created output name {outputName}")

    os.makedirs(args.outputDir, exist_ok = True)

    haibrid_chess_utils.printWithDate(f"Starting read")
    with bz2.open(args.input, 'rt') as f:
        df = pandas.read_csv(f, usecols = target_columns)

    haibrid_chess_utils.printWithDate(f"Starting write")
    with bz2.open(outputName, 'wt') as f:
        df.to_csv(f, index = False)


if __name__ == '__main__':
    main()
