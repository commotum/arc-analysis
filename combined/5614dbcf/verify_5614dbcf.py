import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5614dbcf(I: Grid) -> Grid:
    x0 = canvas(ZERO, THREE_BY_THREE)
    x1 = asindices(x0)
    x2 = shape(I)
    x3 = divide(x2, THREE)
    x4 = first(x3)
    x5 = last(x3)
    x6 = interval(ZERO, x4, ONE)
    x7 = interval(ZERO, x5, ONE)
    x8 = product(x6, x7)
    x9 = rbind(multiply, THREE)
    x10 = apply(x9, x8)
    x11 = matcher(first, FIVE)
    x12 = compose(flip, x11)
    x13 = rbind(sfilter, x12)
    x14 = rbind(toobject, I)
    x15 = lbind(shift, x1)
    x16 = chain(x13, x14, x15)
    x17 = compose(color, x16)
    x18 = lbind(shift, x1)
    x19 = fork(recolor, x17, x18)
    x20 = mapply(x19, x10)
    x21 = paint(I, x20)
    x22 = downscale(x21, THREE)
    return x22