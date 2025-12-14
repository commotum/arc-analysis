import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ff28f65a(I: Grid) -> Grid:
    x0 = palette(I)
    x1 = remove(TWO, x0)
    x2 = lbind(colorcount, I)
    x3 = argmax(x1, x2)
    x4 = shape(I)
    x5 = canvas(x3, x4)
    x6 = hconcat(I, x5)
    x7 = objects(x6, T, F, T)
    x8 = colorfilter(x7, TWO)
    x9 = size(x8)
    x10 = double(x9)
    x11 = interval(ZERO, x10, TWO)
    x12 = apply(tojvec, x11)
    x13 = astuple(ONE, NINE)
    x14 = canvas(x3, x13)
    x15 = fill(x14, ONE, x12)
    x16 = hsplit(x15, THREE)
    x17 = merge(x16)
    return x17