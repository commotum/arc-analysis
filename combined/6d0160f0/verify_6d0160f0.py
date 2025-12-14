import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6d0160f0(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = mostcolor(x1)
    x3 = shape(I)
    x4 = canvas(NEG_ONE, x3)
    x5 = hconcat(I, x4)
    x6 = fill(x5, NEG_ONE, x1)
    x7 = objects(x6, F, F, T)
    x8 = lbind(contained, FOUR)
    x9 = compose(x8, palette)
    x10 = extract(x7, x9)
    x11 = lbind(sfilter, x7)
    x12 = compose(size, x11)
    x13 = rbind(compose, palette)
    x14 = lbind(lbind, contained)
    x15 = chain(x12, x13, x14)
    x16 = merge(x7)
    x17 = palette(I)
    x18 = remove(x2, x17)
    x19 = valmax(x18, x15)
    x20 = matcher(x15, x19)
    x21 = sfilter(x18, x20)
    x22 = lbind(colorcount, x16)
    x23 = argmax(x21, x22)
    x24 = shape(I)
    x25 = canvas(x23, x24)
    x26 = paint(x25, x1)
    x27 = normalize(x10)
    x28 = matcher(first, x2)
    x29 = compose(flip, x28)
    x30 = sfilter(x27, x29)
    x31 = shape(x27)
    x32 = increment(x31)
    x33 = matcher(first, FOUR)
    x34 = sfilter(x27, x33)
    x35 = center(x34)
    x36 = multiply(x32, x35)
    x37 = shift(x30, x36)
    x38 = paint(x26, x37)
    return x38