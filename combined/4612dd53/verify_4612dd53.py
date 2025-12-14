import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_4612dd53(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = box(x1)
    x3 = fill(I, TWO, x2)
    x4 = subgrid(x1, x3)
    x5 = ofcolor(x4, x0)
    x6 = mapply(vfrontier, x5)
    x7 = mapply(hfrontier, x5)
    x8 = size(x6)
    x9 = size(x7)
    x10 = greater(x8, x9)
    x11 = branch(x10, x7, x6)
    x12 = fill(x4, TWO, x11)
    x13 = ofcolor(x12, TWO)
    x14 = ulcorner(x1)
    x15 = shift(x13, x14)
    x16 = underfill(I, TWO, x15)
    return x16