import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_95990924(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = apply(ulcorner, x0)
    x2 = apply(urcorner, x0)
    x3 = apply(llcorner, x0)
    x4 = apply(lrcorner, x0)
    x5 = shift(x1, NEG_UNITY)
    x6 = shift(x2, UP_RIGHT)
    x7 = shift(x3, DOWN_LEFT)
    x8 = shift(x4, UNITY)
    x9 = fill(I, ONE, x5)
    x10 = fill(x9, TWO, x6)
    x11 = fill(x10, THREE, x7)
    x12 = fill(x11, FOUR, x8)
    return x12