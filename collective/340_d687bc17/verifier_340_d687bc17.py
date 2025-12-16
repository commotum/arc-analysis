import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d687bc17(I: Grid) -> Grid:
    x0 = trim(I)
    x1 = asobject(x0)
    x2 = shift(x1, UNITY)
    x3 = apply(initset, x2)
    x4 = toindices(x2)
    x5 = asindices(I)
    x6 = corners(x5)
    x7 = combine(x4, x6)
    x8 = fill(I, NEG_ONE, x7)
    x9 = fgpartition(x8)
    x10 = asindices(I)
    x11 = corners(x10)
    x12 = toobject(x11, I)
    x13 = combine(x2, x12)
    x14 = mostcolor(x13)
    x15 = fill(x8, x14, x7)
    x16 = apply(color, x9)
    x17 = rbind(contained, x16)
    x18 = compose(x17, color)
    x19 = sfilter(x3, x18)
    x20 = lbind(colorfilter, x9)
    x21 = chain(first, x20, color)
    x22 = fork(gravitate, identity, x21)
    x23 = fork(shift, identity, x22)
    x24 = mapply(x23, x19)
    x25 = paint(x15, x24)
    return x25