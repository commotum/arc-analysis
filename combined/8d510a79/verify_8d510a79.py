import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_8d510a79(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = totuple(x0)
    x2 = apply(color, x1)
    x3 = leastcommon(x2)
    x4 = frontiers(I)
    x5 = colorfilter(x4, x3)
    x6 = size(x5)
    x7 = positive(x6)
    x8 = branch(x7, dmirror, identity)
    x9 = ofcolor(I, x3)
    x10 = ofcolor(I, TWO)
    x11 = ofcolor(I, ONE)
    x12 = rbind(gravitate, x9)
    x13 = compose(x12, initset)
    x14 = fork(add, identity, x13)
    x15 = fork(connect, identity, x14)
    x16 = shape(I)
    x17 = maximum(x16)
    x18 = lbind(multiply, x17)
    x19 = lbind(gravitate, x9)
    x20 = chain(x18, sign, x19)
    x21 = compose(x20, initset)
    x22 = fork(add, identity, x21)
    x23 = fork(connect, identity, x22)
    x24 = mapply(x15, x10)
    x25 = mapply(x23, x11)
    x26 = fill(I, TWO, x24)
    x27 = fill(x26, ONE, x25)
    return x27