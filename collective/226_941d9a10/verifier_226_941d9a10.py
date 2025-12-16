import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_941d9a10(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = corners(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = objects(I, T, T, F)
    x5 = colorfilter(x4, x3)
    x6 = fork(add, leftmost, uppermost)
    x7 = argmin(x5, x6)
    x8 = argmax(x5, x6)
    x9 = lbind(sfilter, x5)
    x10 = rbind(compose, leftmost)
    x11 = chain(size, x9, x10)
    x12 = lbind(sfilter, x5)
    x13 = rbind(compose, uppermost)
    x14 = chain(size, x12, x13)
    x15 = lbind(lbind, greater)
    x16 = chain(x11, x15, leftmost)
    x17 = lbind(rbind, greater)
    x18 = chain(x11, x17, leftmost)
    x19 = lbind(lbind, greater)
    x20 = chain(x14, x19, uppermost)
    x21 = lbind(rbind, greater)
    x22 = chain(x14, x21, uppermost)
    x23 = fork(equality, x16, x18)
    x24 = fork(equality, x20, x22)
    x25 = fork(both, x23, x24)
    x26 = extract(x5, x25)
    x27 = fill(I, ONE, x7)
    x28 = fill(x27, THREE, x8)
    x29 = fill(x28, TWO, x26)
    return x29