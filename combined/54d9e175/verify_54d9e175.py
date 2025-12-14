import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_54d9e175(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = leastcolor(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = objects(x5, F, F, T)
    x7 = power(increment, FIVE)
    x8 = lbind(remove, FIVE)
    x9 = lbind(remove, ZERO)
    x10 = chain(x8, x9, palette)
    x11 = chain(x7, first, x10)
    x12 = fork(recolor, x11, toindices)
    x13 = mapply(x12, x6)
    x14 = paint(I, x13)
    return x14