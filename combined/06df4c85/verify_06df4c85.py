import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_06df4c85(I: Grid) -> Grid:
    x0 = asobject(I)
    x1 = frontiers(I)
    x2 = merge(x1)
    x3 = difference(x0, x2)
    x4 = mostcolor(x3)
    x5 = objects(I, T, F, F)
    x6 = color(x2)
    x7 = matcher(color, x6)
    x8 = matcher(color, x4)
    x9 = fork(either, x7, x8)
    x10 = compose(flip, x9)
    x11 = sfilter(x5, x10)
    x12 = merge(x11)
    x13 = palette(x12)
    x14 = lbind(mfilter, x11)
    x15 = lbind(matcher, color)
    x16 = compose(x14, x15)
    x17 = apply(x16, x13)
    x18 = fork(either, vline, hline)
    x19 = lbind(prapply, connect)
    x20 = fork(x19, identity, identity)
    x21 = compose(x20, toindices)
    x22 = rbind(sfilter, x18)
    x23 = chain(merge, x22, x21)
    x24 = fork(recolor, color, x23)
    x25 = mapply(x24, x17)
    x26 = paint(I, x25)
    x27 = paint(x26, x2)
    return x27