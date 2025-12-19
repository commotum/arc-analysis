import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_228f6490(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = objects(I, T, T, F)
    x2 = colorfilter(x1, x0)
    x3 = compose(normalize, toindices)
    x4 = difference(x1, x2)
    x5 = rbind(bordering, I)
    x6 = compose(flip, x5)
    x7 = sfilter(x2, x6)
    x8 = rbind(toobject, I)
    x9 = lbind(mapply, neighbors)
    x10 = compose(x9, toindices)
    x11 = fork(difference, x10, identity)
    x12 = chain(mostcolor, x8, x11)
    x13 = totuple(x7)
    x14 = apply(x12, x13)
    x15 = mostcommon(x14)
    x16 = matcher(x12, x15)
    x17 = sfilter(x7, x16)
    x18 = lbind(argmax, x4)
    x19 = lbind(matcher, x3)
    x20 = chain(x18, x19, x3)
    x21 = compose(color, x20)
    x22 = fork(recolor, x21, identity)
    x23 = mapply(x20, x17)
    x24 = cover(I, x23)
    x25 = mapply(x22, x17)
    x26 = paint(x24, x25)
    return x26