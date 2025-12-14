import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_0dfd9992(I: Grid) -> Grid:
    x0 = palette(I)
    x1 = objects(I, T, F, F)
    x2 = lbind(colorfilter, x1)
    x3 = compose(size, x2)
    x4 = valmin(x0, x3)
    x5 = matcher(x3, x4)
    x6 = sfilter(x0, x5)
    x7 = lbind(colorcount, I)
    x8 = argmin(x6, x7)
    x9 = asobject(I)
    x10 = matcher(first, x8)
    x11 = compose(flip, x10)
    x12 = sfilter(x9, x11)
    x13 = lbind(contained, x8)
    x14 = compose(flip, x13)
    x15 = sfilter(I, x14)
    x16 = asobject(x15)
    x17 = hperiod(x16)
    x18 = dmirror(I)
    x19 = sfilter(x18, x14)
    x20 = asobject(x19)
    x21 = hperiod(x20)
    x22 = astuple(x21, x17)
    x23 = lbind(multiply, x22)
    x24 = neighbors(ORIGIN)
    x25 = mapply(neighbors, x24)
    x26 = apply(x23, x25)
    x27 = lbind(shift, x12)
    x28 = mapply(x27, x26)
    x29 = paint(I, x28)
    return x29