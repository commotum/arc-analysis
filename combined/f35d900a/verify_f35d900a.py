import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f35d900a(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = palette(I)
    x2 = mostcolor(I)
    x3 = remove(x2, x1)
    x4 = lbind(other, x3)
    x5 = compose(x4, color)
    x6 = fork(recolor, x5, outbox)
    x7 = mapply(x6, x0)
    x8 = mapply(toindices, x0)
    x9 = box(x8)
    x10 = difference(x9, x8)
    x11 = lbind(argmin, x8)
    x12 = rbind(compose, initset)
    x13 = lbind(rbind, manhattan)
    x14 = chain(x12, x13, initset)
    x15 = chain(initset, x11, x14)
    x16 = fork(manhattan, initset, x15)
    x17 = compose(even, x16)
    x18 = sfilter(x10, x17)
    x19 = paint(I, x7)
    x20 = fill(x19, FIVE, x18)
    return x20