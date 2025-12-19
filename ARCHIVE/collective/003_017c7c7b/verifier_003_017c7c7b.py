import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_017c7c7b(I: Grid) -> Grid:
    x0 = palette(I)
    x1 = other(x0, ZERO)
    x2 = ofcolor(I, x1)
    x3 = asobject(I)
    x4 = vperiod(x3)
    x5 = height(I)
    x6 = halve(x5)
    x7 = add(x5, x6)
    x8 = width(I)
    x9 = astuple(x7, x8)
    x10 = canvas(ZERO, x9)
    x11 = increment(x7)
    x12 = interval(ZERO, x11, x4)
    x13 = lbind(shift, x2)
    x14 = apply(toivec, x12)
    x15 = mapply(x13, x14)
    x16 = fill(x10, TWO, x15)
    return x16