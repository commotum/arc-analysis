import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_91413438(I: Grid) -> Grid:
    x0 = palette(I)
    x1 = other(x0, ZERO)
    x2 = colorcount(I, x1)
    x3 = colorcount(I, ZERO)
    x4 = dmirror(I)
    x5 = repeat(x4, x2)
    x6 = dmirror(I)
    x7 = shape(x6)
    x8 = canvas(ZERO, x7)
    x9 = multiply(x3, x3)
    x10 = subtract(x9, x2)
    x11 = repeat(x8, x10)
    x12 = combine(x5, x11)
    x13 = merge(x12)
    x14 = dmirror(x13)
    x15 = hsplit(x14, x3)
    x16 = merge(x15)
    return x16