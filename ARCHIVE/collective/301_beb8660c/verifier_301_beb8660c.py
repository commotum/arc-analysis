import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_beb8660c(I: Grid) -> Grid:
    x0 = astuple(identity, rot90)
    x1 = astuple(rot180, rot270)
    x2 = combine(x0, x1)
    x3 = astuple(identity, rot270)
    x4 = astuple(rot180, rot90)
    x5 = combine(x3, x4)
    x6 = pair(x2, x5)
    x7 = rbind(rapply, I)
    x8 = compose(initset, first)
    x9 = chain(first, x7, x8)
    x10 = rbind(ofcolor, EIGHT)
    x11 = chain(lowermost, x10, x9)
    x12 = matcher(x11, ZERO)
    x13 = extract(x6, x12)
    x14 = first(x13)
    x15 = last(x13)
    x16 = x14(I)
    x17 = rot180(x16)
    x18 = shape(x17)
    x19 = lbind(apply, first)
    x20 = lbind(ofcolor, x17)
    x21 = chain(size, x19, x20)
    x22 = palette(I)
    x23 = argmax(x22, x21)
    x24 = partition(x17)
    x25 = matcher(color, x23)
    x26 = compose(flip, x25)
    x27 = sfilter(x24, x26)
    x28 = compose(invert, size)
    x29 = order(x27, x28)
    x30 = apply(normalize, x29)
    x31 = size(x30)
    x32 = interval(ZERO, x31, ONE)
    x33 = apply(toivec, x32)
    x34 = mpapply(shift, x30, x33)
    x35 = canvas(x23, x18)
    x36 = paint(x35, x34)
    x37 = x15(x36)
    return x37