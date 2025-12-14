import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b782dc8a(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = palette(I)
    x2 = remove(x0, x1)
    x3 = lbind(colorcount, I)
    x4 = argmin(x2, x3)
    x5 = ofcolor(I, x0)
    x6 = ofcolor(I, x4)
    x7 = combine(x5, x6)
    x8 = mapply(neighbors, x7)
    x9 = difference(x8, x7)
    x10 = toobject(x9, I)
    x11 = leastcolor(x10)
    x12 = ofcolor(I, x0)
    x13 = first(x12)
    x14 = initset(x13)
    x15 = objects(I, T, F, F)
    x16 = colorfilter(x15, x11)
    x17 = lbind(adjacent, x7)
    x18 = mfilter(x16, x17)
    x19 = toindices(x18)
    x20 = rbind(manhattan, x14)
    x21 = chain(even, x20, initset)
    x22 = sfilter(x19, x21)
    x23 = fill(I, x4, x19)
    x24 = fill(x23, x0, x22)
    return x24