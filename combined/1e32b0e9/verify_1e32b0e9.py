import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1e32b0e9(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = totuple(x0)
    x2 = apply(color, x1)
    x3 = leastcommon(x2)
    x4 = matcher(color, x3)
    x5 = sfilter(x0, x4)
    x6 = merge(x5)
    x7 = color(x6)
    x8 = shape(I)
    x9 = canvas(x7, x8)
    x10 = hconcat(I, x9)
    x11 = objects(x10, F, T, T)
    x12 = first(x11)
    x13 = box(x12)
    x14 = rbind(contained, x13)
    x15 = compose(x14, last)
    x16 = sfilter(x12, x15)
    x17 = color(x16)
    x18 = palette(I)
    x19 = remove(x7, x18)
    x20 = other(x19, x17)
    x21 = rbind(colorcount, x17)
    x22 = argmin(x11, x21)
    x23 = apply(ulcorner, x11)
    x24 = normalize(x22)
    x25 = matcher(first, x20)
    x26 = sfilter(x24, x25)
    x27 = toindices(x26)
    x28 = lbind(shift, x27)
    x29 = mapply(x28, x23)
    x30 = ofcolor(I, x20)
    x31 = difference(x29, x30)
    x32 = fill(I, x7, x31)
    return x32