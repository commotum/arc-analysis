import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_855e0971(I: Grid) -> Grid:
    x0 = lbind(greater, THREE)
    x1 = chain(x0, size, dedupe)
    x2 = apply(x1, I)
    x3 = contained(F, x2)
    x4 = flip(x3)
    x5 = branch(x4, identity, dmirror)
    x6 = x5(I)
    x7 = rbind(toobject, I)
    x8 = chain(palette, x7, neighbors)
    x9 = lbind(chain, flip)
    x10 = rbind(x9, x8)
    x11 = lbind(lbind, contained)
    x12 = compose(x10, x11)
    x13 = lbind(ofcolor, I)
    x14 = fork(sfilter, x13, x12)
    x15 = compose(size, x14)
    x16 = palette(I)
    x17 = argmax(x16, x15)
    x18 = objects(x6, T, T, F)
    x19 = colorfilter(x18, x17)
    x20 = difference(x18, x19)
    x21 = rbind(subgrid, x6)
    x22 = order(x20, uppermost)
    x23 = apply(x21, x22)
    x24 = lbind(recolor, x17)
    x25 = lbind(mapply, vfrontier)
    x26 = rbind(ofcolor, x17)
    x27 = chain(x24, x25, x26)
    x28 = fork(paint, identity, x27)
    x29 = mapply(x28, x23)
    x30 = x5(x29)
    return x30