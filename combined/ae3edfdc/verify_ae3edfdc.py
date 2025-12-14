import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ae3edfdc(I: Grid) -> Grid:
    x0 = ofcolor(I, ONE)
    x1 = center(x0)
    x2 = ofcolor(I, TWO)
    x3 = center(x2)
    x4 = ofcolor(I, THREE)
    x5 = ofcolor(I, SEVEN)
    x6 = lbind(add, x1)
    x7 = initset(x1)
    x8 = rbind(position, x7)
    x9 = compose(invert, x8)
    x10 = chain(x6, x9, initset)
    x11 = lbind(add, x3)
    x12 = initset(x3)
    x13 = rbind(position, x12)
    x14 = compose(invert, x13)
    x15 = chain(x11, x14, initset)
    x16 = apply(x10, x5)
    x17 = apply(x15, x4)
    x18 = combine(x4, x5)
    x19 = cover(I, x18)
    x20 = fill(x19, SEVEN, x16)
    x21 = fill(x20, THREE, x17)
    return x21