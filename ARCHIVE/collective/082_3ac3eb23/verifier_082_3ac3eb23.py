import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3ac3eb23(I: Grid) -> Grid:
    x0 = astuple(identity, dmirror)
    x1 = astuple(cmirror, hmirror)
    x2 = combine(x0, x1)
    x3 = chain(lowermost, merge, fgpartition)
    x4 = rbind(rapply, I)
    x5 = lbind(compose, x3)
    x6 = compose(initset, x5)
    x7 = chain(first, x4, x6)
    x8 = matcher(x7, ZERO)
    x9 = extract(x2, x8)
    x10 = x9(I)
    x11 = objects(x10, T, F, T)
    x12 = height(x10)
    x13 = interval(ZERO, x12, TWO)
    x14 = height(x10)
    x15 = interval(ONE, x14, TWO)
    x16 = rbind(apply, x13)
    x17 = lbind(rbind, astuple)
    x18 = chain(x16, x17, last)
    x19 = rbind(apply, x15)
    x20 = lbind(rbind, astuple)
    x21 = compose(increment, last)
    x22 = chain(x19, x20, x21)
    x23 = rbind(apply, x15)
    x24 = lbind(rbind, astuple)
    x25 = compose(decrement, last)
    x26 = chain(x23, x24, x25)
    x27 = fork(combine, x18, x22)
    x28 = fork(combine, x27, x26)
    x29 = compose(x28, center)
    x30 = fork(recolor, color, x29)
    x31 = mapply(x30, x11)
    x32 = paint(x10, x31)
    x33 = x9(x32)
    return x33