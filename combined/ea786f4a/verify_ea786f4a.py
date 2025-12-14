import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ea786f4a(I: Grid) -> Grid:
    x0 = shape(I)
    x1 = halve(x0)
    x2 = rbind(shoot, UP_RIGHT)
    x3 = rbind(shoot, DOWN_LEFT)
    x4 = fork(combine, x2, x3)
    x5 = rbind(shoot, UNITY)
    x6 = rbind(shoot, NEG_UNITY)
    x7 = fork(combine, x5, x6)
    x8 = fork(combine, x4, x7)
    x9 = index(I, x1)
    x10 = x8(x1)
    x11 = fill(I, x9, x10)
    return x11