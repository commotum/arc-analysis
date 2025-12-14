import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_834ec97d(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = argmin(x0, size)
    x2 = cover(I, x1)
    x3 = shift(x1, DOWN)
    x4 = paint(x2, x3)
    x5 = leftmost(x1)
    x6 = width(I)
    x7 = interval(x5, x6, TWO)
    x8 = leftmost(x1)
    x9 = interval(x8, NEG_ONE, NEG_TWO)
    x10 = combine(x7, x9)
    x11 = rbind(shoot, UP)
    x12 = uppermost(x1)
    x13 = lbind(astuple, x12)
    x14 = apply(x13, x10)
    x15 = mapply(x11, x14)
    x16 = fill(x4, FOUR, x15)
    return x16