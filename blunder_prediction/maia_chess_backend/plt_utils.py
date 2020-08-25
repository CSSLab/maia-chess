import matplotlib.pyplot as plt
import os
import os.path
import seaborn

def multi_savefig(save_name, dir_name = 'images', save_types = ('pdf', 'png', 'svg')):
    os.makedirs(dir_name, exist_ok = True)
    for sType in save_types:
        dName = os.path.join(dir_name, sType)
        os.makedirs(dName, exist_ok = True)

        fname = f'{save_name}.{sType}'

        plt.savefig(os.path.join(dName, fname), format = sType, dpi = 300, transparent = True)

def plot_pieces(board_a):
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize = (16, 10))
    axiter = iter(axes.flatten())
    for i in range(17):
        seaborn.heatmap(board_a[i], ax = next(axiter), cbar = False, vmin=0, vmax=1, square = True)

    axes[-1,-1].set_axis_off()
    for i, n in enumerate(['Knights', 'Bishops', 'Rooks','Queen', 'King']):
        axes[0,i + 1].set_title(n)
        axes[1,i + 1].set_title(n)
    axes[0,0].set_title('Active Player Pieces\nPawns')
    axes[1,0].set_title('Opponent Pieces\nPawns')
    axes[2,0].set_title('Other Values\n Is White')
    for i in range(4):
        axes[2,i + 1].set_title('Castling')
