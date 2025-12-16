import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_97999447(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = asobject(I)
    x2 = matcher(first, x0)
    x3 = compose(flip, x2)
    x4 = sfilter(x1, x3)
    x5 = apply(initset, x4)
    x6 = apply(toindices, x5)
    x7 = rbind(shoot, RIGHT)
    x8 = compose(x7, center)
    x9 = fork(recolor, color, x8)
    x10 = mapply(x9, x5)
    x11 = paint(I, x10)
    x12 = width(I)
    x13 = interval(ZERO, x12, ONE)
    x14 = apply(double, x13)
    x15 = apply(increment, x14)
    x16 = apply(tojvec, x15)
    x17 = prapply(shift, x6, x16)
    x18 = merge(x17)
    x19 = fill(x11, FIVE, x18)
    return x19