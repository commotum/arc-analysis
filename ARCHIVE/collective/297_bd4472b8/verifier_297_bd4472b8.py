import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_bd4472b8(I: Grid) -> Grid:
    x0 = compose(positive, size)
    x1 = rbind(sfilter, hline)
    x2 = chain(x0, x1, frontiers)
    x3 = chain(size, dedupe, first)
    x4 = chain(size, dedupe, last)
    x5 = fork(greater, x3, x4)
    x6 = fork(both, x2, x5)
    x7 = astuple(identity, rot90)
    x8 = astuple(rot180, rot270)
    x9 = combine(x7, x8)
    x10 = astuple(identity, rot270)
    x11 = astuple(rot180, rot90)
    x12 = combine(x10, x11)
    x13 = pair(x9, x12)
    x14 = rbind(rapply, I)
    x15 = compose(initset, first)
    x16 = chain(first, x14, x15)
    x17 = compose(x6, x16)
    x18 = extract(x13, x17)
    x19 = first(x18)
    x20 = last(x18)
    x21 = x19(I)
    x22 = first(x21)
    x23 = repeat(x22, ONE)
    x24 = dmirror(x23)
    x25 = width(x21)
    x26 = hupscale(x24, x25)
    x27 = asobject(x26)
    x28 = height(x21)
    x29 = height(x27)
    x30 = interval(ZERO, x28, x29)
    x31 = lbind(shift, x27)
    x32 = apply(toivec, x30)
    x33 = mapply(x31, x32)
    x34 = shift(x33, TWO_BY_ZERO)
    x35 = paint(x21, x34)
    x36 = x20(x35)
    return x36