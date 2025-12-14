import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_85c4e7cd(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = box(x0)
    x2 = shape(I)
    x3 = minimum(x2)
    x4 = halve(x3)
    x5 = interval(ONE, x4, ONE)
    x6 = lbind(power, inbox)
    x7 = rbind(rapply, x1)
    x8 = compose(initset, x6)
    x9 = chain(first, x7, x8)
    x10 = apply(x9, x5)
    x11 = repeat(x1, ONE)
    x12 = combine(x11, x10)
    x13 = rbind(toobject, I)
    x14 = compose(color, x13)
    x15 = apply(x14, x12)
    x16 = interval(ZERO, x4, ONE)
    x17 = pair(x16, x15)
    x18 = compose(invert, first)
    x19 = order(x17, x18)
    x20 = apply(last, x19)
    x21 = mpapply(recolor, x20, x12)
    x22 = paint(I, x21)
    return x22