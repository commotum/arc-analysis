import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_29623171(I: Grid) -> Grid:
    x0 = compress(I)
    x1 = leastcolor(x0)
    x2 = mostcolor(x0)
    x3 = frontiers(I)
    x4 = sfilter(x3, hline)
    x5 = size(x4)
    x6 = increment(x5)
    x7 = sfilter(x3, vline)
    x8 = size(x7)
    x9 = increment(x8)
    x10 = height(I)
    x11 = decrement(x6)
    x12 = subtract(x10, x11)
    x13 = divide(x12, x6)
    x14 = width(I)
    x15 = decrement(x9)
    x16 = subtract(x14, x15)
    x17 = divide(x16, x9)
    x18 = astuple(x13, x17)
    x19 = canvas(ZERO, x18)
    x20 = asindices(x19)
    x21 = astuple(x6, x9)
    x22 = canvas(ZERO, x21)
    x23 = asindices(x22)
    x24 = astuple(x13, x17)
    x25 = increment(x24)
    x26 = rbind(multiply, x25)
    x27 = apply(x26, x23)
    x28 = rbind(toobject, I)
    x29 = lbind(shift, x20)
    x30 = compose(x28, x29)
    x31 = apply(x30, x27)
    x32 = rbind(colorcount, x1)
    x33 = valmax(x31, x32)
    x34 = rbind(colorcount, x1)
    x35 = matcher(x34, x33)
    x36 = mfilter(x31, x35)
    x37 = replace(I, x1, x2)
    x38 = fill(x37, x1, x36)
    return x38