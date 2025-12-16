import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_bc1d5164(I: Grid) -> Grid:
    x0 = height(I)
    x1 = halve(x0)
    x2 = increment(x1)
    x3 = width(I)
    x4 = halve(x3)
    x5 = frontiers(I)
    x6 = merge(x5)
    x7 = mostcolor(x6)
    x8 = astuple(x2, x4)
    x9 = canvas(x7, x8)
    x10 = asindices(x9)
    x11 = toobject(x10, I)
    x12 = increment(x4)
    x13 = tojvec(x12)
    x14 = shift(x10, x13)
    x15 = toobject(x14, I)
    x16 = decrement(x2)
    x17 = toivec(x16)
    x18 = shift(x10, x17)
    x19 = toobject(x18, I)
    x20 = decrement(x2)
    x21 = increment(x4)
    x22 = astuple(x20, x21)
    x23 = shift(x10, x22)
    x24 = toobject(x23, I)
    x25 = palette(I)
    x26 = other(x25, x7)
    x27 = matcher(first, x26)
    x28 = rbind(sfilter, x27)
    x29 = chain(toindices, x28, normalize)
    x30 = x29(x11)
    x31 = x29(x15)
    x32 = x29(x19)
    x33 = x29(x24)
    x34 = combine(x30, x31)
    x35 = combine(x32, x33)
    x36 = combine(x34, x35)
    x37 = fill(x9, x26, x36)
    return x37