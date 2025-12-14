import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_bdad9b1f(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = sfilter(x0, hline)
    x2 = sfilter(x0, vline)
    x3 = compose(hfrontier, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x1)
    x6 = compose(vfrontier, center)
    x7 = fork(recolor, color, x6)
    x8 = mapply(x7, x2)
    x9 = combine(x5, x8)
    x10 = paint(I, x9)
    x11 = toindices(x5)
    x12 = toindices(x8)
    x13 = intersection(x11, x12)
    x14 = fill(x10, FOUR, x13)
    return x14