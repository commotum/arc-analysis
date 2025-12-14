import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_2c608aff(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(equality, toindices, backdrop)
    x2 = sfilter(x0, x1)
    x3 = argmax(x2, size)
    x4 = color(x3)
    x5 = palette(I)
    x6 = remove(x4, x5)
    x7 = lbind(colorcount, I)
    x8 = argmin(x6, x7)
    x9 = toindices(x3)
    x10 = apply(first, x9)
    x11 = toindices(x3)
    x12 = apply(last, x11)
    x13 = rbind(contained, x10)
    x14 = compose(x13, first)
    x15 = rbind(contained, x12)
    x16 = compose(x15, last)
    x17 = fork(either, x14, x16)
    x18 = ofcolor(I, x8)
    x19 = sfilter(x18, x17)
    x20 = rbind(gravitate, x3)
    x21 = compose(x20, initset)
    x22 = fork(add, identity, x21)
    x23 = fork(connect, identity, x22)
    x24 = mapply(x23, x19)
    x25 = fill(I, x8, x24)
    return x25