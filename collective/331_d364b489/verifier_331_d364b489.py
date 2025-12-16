import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d364b489(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = toindices(x1)
    x3 = shift(x2, DOWN)
    x4 = fill(I, EIGHT, x3)
    x5 = shift(x2, UP)
    x6 = fill(x4, TWO, x5)
    x7 = shift(x2, RIGHT)
    x8 = fill(x6, SIX, x7)
    x9 = shift(x2, LEFT)
    x10 = fill(x8, SEVEN, x9)
    return x10