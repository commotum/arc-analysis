import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f8b3ba0a(I: Grid) -> Grid:
    x0 = compress(I)
    x1 = astuple(THREE, ONE)
    x2 = palette(x0)
    x3 = lbind(colorcount, x0)
    x4 = compose(invert, x3)
    x5 = order(x2, x4)
    x6 = rbind(canvas, UNITY)
    x7 = apply(x6, x5)
    x8 = merge(x7)
    x9 = size(x2)
    x10 = decrement(x9)
    x11 = astuple(x10, ONE)
    x12 = crop(x8, DOWN, x11)
    return x12