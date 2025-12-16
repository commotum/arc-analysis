import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_88a10436(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = argmax(x0, size)
    x2 = normalize(x1)
    x3 = shape(x1)
    x4 = halve(x3)
    x5 = invert(x4)
    x6 = shift(x2, x5)
    x7 = sizefilter(x0, ONE)
    x8 = apply(center, x7)
    x9 = lbind(shift, x6)
    x10 = mapply(x9, x8)
    x11 = paint(I, x10)
    return x11