import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_53b68214(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = mostcolor(I)
    x3 = width(I)
    x4 = astuple(TEN, x3)
    x5 = canvas(x2, x4)
    x6 = interval(ONE, SIX, ONE)
    x7 = invert(TEN)
    x8 = interval(x7, TEN, ONE)
    x9 = product(x6, x8)
    x10 = remove(ORIGIN, x9)
    x11 = lbind(intersection, x1)
    x12 = lbind(shift, x1)
    x13 = compose(x11, x12)
    x14 = toindices(x1)
    x15 = lbind(intersection, x14)
    x16 = lbind(shift, x14)
    x17 = compose(x15, x16)
    x18 = compose(size, x13)
    x19 = compose(size, x17)
    x20 = fork(equality, x18, x19)
    x21 = chain(positive, size, x13)
    x22 = fork(both, x20, x21)
    x23 = sfilter(x10, x22)
    x24 = compose(size, x13)
    x25 = valmax(x23, x24)
    x26 = compose(size, x13)
    x27 = matcher(x26, x25)
    x28 = sfilter(x23, x27)
    x29 = fork(multiply, first, last)
    x30 = argmax(x28, x29)
    x31 = interval(ZERO, TEN, ONE)
    x32 = lbind(shift, x1)
    x33 = lbind(multiply, x30)
    x34 = compose(x32, x33)
    x35 = mapply(x34, x31)
    x36 = paint(x5, x35)
    return x36