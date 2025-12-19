import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1a07d186(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = mostcolor(I)
    x2 = asindices(I)
    x3 = ofcolor(I, x1)
    x4 = difference(x2, x3)
    x5 = mapply(toindices, x0)
    x6 = difference(x4, x5)
    x7 = toobject(x6, I)
    x8 = apply(initset, x7)
    x9 = fill(I, x1, x6)
    x10 = lbind(fork, shift)
    x11 = lbind(x10, identity)
    x12 = lbind(rbind, gravitate)
    x13 = compose(x11, x12)
    x14 = lbind(colorfilter, x8)
    x15 = compose(x14, color)
    x16 = fork(mapply, x13, x15)
    x17 = mapply(x16, x0)
    x18 = paint(x9, x17)
    return x18