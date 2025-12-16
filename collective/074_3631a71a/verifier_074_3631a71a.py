import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3631a71a(I: Grid) -> Grid:
    x0 = lbind(compose, flip)
    x1 = lbind(matcher, first)
    x2 = compose(x0, x1)
    x3 = rbind(compose, asobject)
    x4 = lbind(rbind, sfilter)
    x5 = chain(x3, x4, x2)
    x6 = rbind(shift, ORIGIN)
    x7 = compose(x6, dmirror)
    x8 = rbind(shift, TWO_BY_TWO)
    x9 = compose(x8, cmirror)
    x10 = rbind(shift, TWO_BY_ZERO)
    x11 = compose(x10, hmirror)
    x12 = rbind(shift, ZERO_BY_TWO)
    x13 = compose(x12, vmirror)
    x14 = lbind(fork, paint)
    x15 = lbind(x14, identity)
    x16 = lbind(compose, x7)
    x17 = chain(x15, x16, x5)
    x18 = lbind(compose, x9)
    x19 = chain(x15, x18, x5)
    x20 = lbind(compose, x11)
    x21 = chain(x15, x20, x5)
    x22 = lbind(compose, x13)
    x23 = chain(x15, x22, x5)
    x24 = rbind(rapply, I)
    x25 = chain(first, x24, initset)
    x26 = fork(compose, x23, x21)
    x27 = fork(compose, x19, x17)
    x28 = fork(compose, x26, x27)
    x29 = compose(x25, x28)
    x30 = palette(I)
    x31 = fork(equality, identity, dmirror)
    x32 = compose(x31, x29)
    x33 = argmax(x30, x32)
    x34 = x29(x33)
    return x34