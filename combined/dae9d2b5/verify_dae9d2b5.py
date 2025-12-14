import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_dae9d2b5(I: Grid) -> Grid:
    x0 = hsplit(I, TWO)
    x1 = apply(numcolors, x0)
    x2 = equality(x1, TWO_BY_TWO)
    x3 = branch(x2, hsplit, vsplit)
    x4 = x3(I, TWO)
    x5 = first(x4)
    x6 = last(x4)
    x7 = palette(x5)
    x8 = palette(x6)
    x9 = intersection(x7, x8)
    x10 = first(x9)
    x11 = palette(x5)
    x12 = other(x11, x10)
    x13 = palette(x6)
    x14 = other(x13, x10)
    x15 = shape(x5)
    x16 = canvas(x10, x15)
    x17 = ofcolor(x5, x12)
    x18 = ofcolor(x6, x14)
    x19 = combine(x17, x18)
    x20 = fill(x16, SIX, x19)
    return x20