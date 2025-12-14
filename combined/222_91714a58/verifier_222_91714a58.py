import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_91714a58(I: Grid) -> Grid:
    x0 = shape(I)
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = argmax(x2, size)
    x4 = mostcolor(x3)
    x5 = mostcolor(I)
    x6 = canvas(x5, x0)
    x7 = paint(x6, x3)
    x8 = mostcolor(I)
    x9 = color(x3)
    x10 = astuple(x8, ORIGIN)
    x11 = astuple(x9, RIGHT)
    x12 = astuple(x8, ZERO_BY_TWO)
    x13 = initset(x12)
    x14 = insert(x11, x13)
    x15 = insert(x10, x14)
    x16 = dmirror(x15)
    x17 = toindices(x15)
    x18 = lbind(shift, x17)
    x19 = occurrences(x7, x15)
    x20 = mapply(x18, x19)
    x21 = toindices(x16)
    x22 = lbind(shift, x21)
    x23 = occurrences(x7, x16)
    x24 = mapply(x22, x23)
    x25 = combine(x20, x24)
    x26 = fill(x7, x8, x25)
    return x26