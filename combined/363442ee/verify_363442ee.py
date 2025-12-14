import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_363442ee(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = mostcolor(I)
    x3 = fill(I, x2, x1)
    x4 = objects(x3, F, F, T)
    x5 = argmax(x4, size)
    x6 = remove(x5, x4)
    x7 = apply(center, x6)
    x8 = normalize(x5)
    x9 = shape(x5)
    x10 = halve(x9)
    x11 = invert(x10)
    x12 = shift(x8, x11)
    x13 = lbind(shift, x12)
    x14 = mapply(x13, x7)
    x15 = paint(I, x14)
    return x15