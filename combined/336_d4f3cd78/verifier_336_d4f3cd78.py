import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d4f3cd78(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = toindices(x1)
    x3 = box(x2)
    x4 = difference(x3, x2)
    x5 = inbox(x2)
    x6 = backdrop(x5)
    x7 = lbind(position, x6)
    x8 = compose(x7, initset)
    x9 = lowermost(x6)
    x10 = rightmost(x6)
    x11 = uppermost(x6)
    x12 = leftmost(x6)
    x13 = rbind(greater, x9)
    x14 = compose(x13, first)
    x15 = lbind(greater, x11)
    x16 = compose(x15, first)
    x17 = rbind(greater, x10)
    x18 = compose(x17, last)
    x19 = lbind(greater, x12)
    x20 = compose(x19, last)
    x21 = compose(invert, x16)
    x22 = fork(add, x14, x21)
    x23 = compose(invert, x20)
    x24 = fork(add, x18, x23)
    x25 = fork(astuple, x22, x24)
    x26 = fork(shoot, identity, x25)
    x27 = mapply(x26, x4)
    x28 = combine(x27, x6)
    x29 = fill(I, EIGHT, x28)
    return x29