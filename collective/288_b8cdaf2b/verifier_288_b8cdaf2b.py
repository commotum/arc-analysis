import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b8cdaf2b(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = astuple(dmirror, cmirror)
    x2 = astuple(hmirror, identity)
    x3 = combine(x1, x2)
    x4 = rbind(rapply, I)
    x5 = chain(first, x4, initset)
    x6 = rbind(ofcolor, x0)
    x7 = chain(lowermost, x6, x5)
    x8 = chain(decrement, height, x5)
    x9 = fork(equality, x7, x8)
    x10 = extract(x3, x9)
    x11 = x10(I)
    x12 = ofcolor(x11, x0)
    x13 = shift(x12, UP)
    x14 = ulcorner(x13)
    x15 = urcorner(x13)
    x16 = shoot(x14, NEG_UNITY)
    x17 = shoot(x15, UP_RIGHT)
    x18 = combine(x16, x17)
    x19 = underfill(x11, x0, x18)
    x20 = x10(x19)
    return x20