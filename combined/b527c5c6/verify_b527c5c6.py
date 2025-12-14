import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b527c5c6(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = lbind(matcher, first)
    x2 = compose(x1, leastcolor)
    x3 = fork(sfilter, identity, x2)
    x4 = compose(center, x3)
    x5 = compose(dneighbors, x4)
    x6 = fork(difference, x5, toindices)
    x7 = compose(first, x6)
    x8 = fork(subtract, x7, x4)
    x9 = compose(invert, x8)
    x10 = fork(shoot, x4, x9)
    x11 = fork(intersection, toindices, x10)
    x12 = chain(decrement, size, x11)
    x13 = fork(shoot, x4, x8)
    x14 = lbind(power, outbox)
    x15 = compose(x14, x12)
    x16 = compose(initset, x15)
    x17 = fork(rapply, x16, x13)
    x18 = chain(backdrop, first, x17)
    x19 = fork(recolor, leastcolor, x13)
    x20 = fork(difference, x18, x13)
    x21 = fork(recolor, mostcolor, x20)
    x22 = fork(combine, x19, x21)
    x23 = mapply(x22, x0)
    x24 = paint(I, x23)
    return x24