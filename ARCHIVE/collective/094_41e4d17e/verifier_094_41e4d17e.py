import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_41e4d17e(I: Grid) -> Grid:
    x0 = lbind(equality, NINE)
    x1 = compose(x0, size)
    x2 = fork(equality, height, width)
    x3 = fork(both, x1, x2)
    x4 = objects(I, T, F, F)
    x5 = sfilter(x4, x3)
    x6 = fork(combine, vfrontier, hfrontier)
    x7 = compose(x6, center)
    x8 = mapply(x7, x5)
    x9 = underfill(I, SIX, x8)
    return x9