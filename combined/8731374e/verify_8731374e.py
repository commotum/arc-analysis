import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_8731374e(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = argmax(x0, size)
    x2 = color(x1)
    x3 = subgrid(x1, I)
    x4 = lbind(insert, DOWN)
    x5 = compose(lrcorner, asindices)
    x6 = chain(x4, initset, x5)
    x7 = fork(subgrid, x6, identity)
    x8 = matcher(identity, x2)
    x9 = rbind(subtract, TWO)
    x10 = rbind(sfilter, x8)
    x11 = compose(x9, width)
    x12 = chain(size, x10, first)
    x13 = fork(greater, x11, x12)
    x14 = rbind(branch, identity)
    x15 = rbind(x14, x7)
    x16 = chain(initset, x15, x13)
    x17 = fork(rapply, x16, identity)
    x18 = compose(first, x17)
    x19 = compose(x18, rot90)
    x20 = double(EIGHT)
    x21 = double(x20)
    x22 = power(x19, x21)
    x23 = x22(x3)
    x24 = leastcolor(x23)
    x25 = ofcolor(x23, x24)
    x26 = fork(combine, vfrontier, hfrontier)
    x27 = mapply(x26, x25)
    x28 = fill(x23, x24, x27)
    return x28