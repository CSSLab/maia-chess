import argparse
import pandas
import os
import os.path
import numpy as np
import haibrid_chess_utils

@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='Make train testr split of grouped boards', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input', help='input CSV')
    parser.add_argument('outputDir', help='output CSVs dir')
    parser.add_argument('--nrows', type=int, help='number of rows to read in, FOR TESTING', default = None)
    parser.add_argument('--blunder_ratio', type=float, help='ratio to declare a positive class', default = .1)
    parser.add_argument('--min_count', type=int, help='ratio to declare a positive class', default = 10)

    args = parser.parse_args()
    os.makedirs(os.path.join(args.outputDir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.outputDir, 'train'), exist_ok=True)

    haibrid_chess_utils.printWithDate(f"Loading: {args.input}")

    df = pandas.read_csv(args.input)

    haibrid_chess_utils.printWithDate(f"Filtering: {args.input}")
    df = df[df['count'] >= args.min_count].copy()

    df['is_blunder_mean'] = df['is_blunder_wr_mean'] > .1
    df['board_extended'] = df['board'].apply(lambda x : x + ' KQkq - 0 1')
    df['white_active'] = df['board'].apply(lambda x : x.endswith('w'))
    df['has_nonblunder_move'] = df['top_nonblunder'].isna() == False
    df['has_blunder_move'] = df['top_blunder'].isna() == False
    df['is_test'] = [np.random.random() < args.blunder_ratio for i in range(len(df))]

    haibrid_chess_utils.printWithDate(f"Wrting to: {args.outputDir}")
    df[df['is_test']].to_csv('/datadrive/group_csv/test/grouped_fens.csv.bz2', compression = 'bz2')

    df[df['is_test'] == False].to_csv(os.path.join(args.outputDir, 'train', 'grouped_fens.csv.bz2'), compression = 'bz2')
    df[df['is_test'] == True].to_csv(os.path.join(args.outputDir, 'test', 'grouped_fens.csv.bz2'), compression = 'bz2')

if __name__ == '__main__':
    main()
