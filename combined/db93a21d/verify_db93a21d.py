import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_db93a21d(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = merge(x0)
    x2 = toindices(x1)
    x3 = rbind(shoot, DOWN)
    x4 = mapply(x3, x2)
    x5 = underfill(I, ONE, x4)
    x6 = lbind(power, outbox)
    x7 = chain(x6, halve, width)
    x8 = initset(x7)
    x9 = lbind(rapply, x8)
    x10 = fork(rapply, x9, identity)
    x11 = compose(first, x10)
    x12 = compose(backdrop, x11)
    x13 = fork(difference, x12, toindices)
    x14 = mapply(x13, x0)
    x15 = mostcolor(I)
    x16 = ofcolor(I, x15)
    x17 = intersection(x14, x16)
    x18 = fill(x5, THREE, x17)
    return x18