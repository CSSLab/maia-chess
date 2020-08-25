#from .uci import *
from .games import *
from .utils import *
from .tourney import *
from .loaders import *
from .models_loader import *
from .logging import *
from .fen_to_vec import *
from .bat_files import *
from .plt_utils import *
from .model_loader import load_model_config
#from .pickle4reducer import *
#from .boardTrees import *
#from .stockfishAnalysis import *

#Tensorflow stuff
try:
    from .tf_process import *
    from .tf_net import *
    from .tf_blocks import *
except ImportError:
    pass

fics_header = [
    'game_id',
    'rated',
    'name',
    'opp_name',
    'elo',
    'oppelo',
    'num_legal_moves',
    'num_blunders',
    'blunder',
    'eval_before_move',
    'eval_after_move',
    'to_move',
    'is_comp',
    'opp_is_comp',
    'time_control',
    'ECO',
    'result',
    'time_left',
    'opp_time_left',
    'time_used',
    'move_idx',
    'move',
    'material',
    'position',
    'stdpos',
    'unkown'
]

__version__ = '1.0.0'
