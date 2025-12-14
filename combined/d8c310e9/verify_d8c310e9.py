import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d8c310e9(I: Grid) -> Grid:
    x0 = astuple(identity, rot90)
    x1 = astuple(rot180, rot270)
    x2 = combine(x0, x1)
    x3 = astuple(identity, rot270)
    x4 = astuple(rot180, rot90)
    x5 = combine(x3, x4)
    x6 = pair(x2, x5)
    x7 = chain(size, dedupe, first)
    x8 = matcher(x7, ONE)
    x9 = compose(first, cmirror)
    x10 = chain(size, dedupe, x9)
    x11 = matcher(x10, ONE)
    x12 = fork(both, x8, x11)
    x13 = rbind(rapply, I)
    x14 = compose(initset, first)
    x15 = chain(first, x13, x14)
    x16 = compose(x12, x15)
    x17 = extract(x6, x16)
    x18 = first(x17)
    x19 = last(x17)
    x20 = x18(I)
    x21 = width(x20)
    x22 = decrement(x21)
    x23 = tojvec(x22)
    x24 = index(x20, x23)
    x25 = asobject(x20)
    x26 = matcher(first, x24)
    x27 = compose(flip, x26)
    x28 = sfilter(x25, x27)
    x29 = hperiod(x28)
    x30 = width(x20)
    x31 = increment(x30)
    x32 = interval(ZERO, x31, x29)
    x33 = apply(tojvec, x32)
    x34 = lbind(shift, x28)
    x35 = mapply(x34, x33)
    x36 = paint(x20, x35)
    x37 = x19(x36)
    return x37