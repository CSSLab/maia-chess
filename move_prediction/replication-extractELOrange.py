import maia_chess_backend

import argparse
import bz2

#@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('eloMin', type=int, help='min ELO')
    parser.add_argument('eloMax', type=int, help='max ELO')
    parser.add_argument('output', help='output file')
    parser.add_argument('targets', nargs='+', help='target files')
    parser.add_argument('--remove_bullet', action='store_true', help='Remove bullet and ultrabullet games')
    parser.add_argument('--remove_low_time', action='store_true', help='Remove low time moves from games')

    args = parser.parse_args()
    gamesWritten = 0
    print(f"Starting writing to: {args.output}")
    with bz2.open(args.output, 'wt') as f:
        for num_files, target in enumerate(sorted(args.targets)):
            print(f"{num_files} reading: {target}")
            Games = maia_chess_backend.LightGamesFile(target, parseMoves = False)
            for i, (dat, lines) in enumerate(Games):
                try:
                    whiteELO = int(dat['WhiteElo'])
                    BlackELO = int(dat['BlackElo'])
                except ValueError:
                    continue
                if whiteELO > args.eloMax or whiteELO <= args.eloMin:
                    continue
                elif BlackELO > args.eloMax or BlackELO <= args.eloMin:
                    continue
                elif dat['Result']  not in ['1-0', '0-1', '1/2-1/2']:
                    continue
                elif args.remove_bullet and 'Bullet' in dat['Event']:
                    continue
                else:
                    if args.remove_low_time:
                        f.write(maia_chess_backend.remove_low_time(lines))
                    else:
                        f.write(lines)
                    gamesWritten += 1
                if i % 1000 == 0:
                    print(f"{i}: written {gamesWritten} files {num_files}: {target}".ljust(79), end = '\r')
            print(f"Done: {target} {i}".ljust(79))

if __name__ == '__main__':
    main()
