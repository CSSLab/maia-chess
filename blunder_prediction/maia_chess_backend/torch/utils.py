import torch

from ..fen_to_vec import fenToVec
from ..utils import fen_extend

def fenToTensor(fenstr):
    try:
        t = torch.from_numpy(fenToVec(fenstr))
    except AttributeError:
        t = torch.from_numpy(fenToVec(fen_extend(fenstr)))
    if torch.cuda.is_available():
        t = t.cuda()
    return t.float()
