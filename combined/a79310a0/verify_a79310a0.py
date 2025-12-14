import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a79310a0(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = mostcolor(I)
    x3 = fill(I, x2, x1)
    x4 = shift(x1, DOWN)
    x5 = fill(x3, TWO, x4)
    return x5