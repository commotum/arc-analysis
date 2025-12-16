import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_cf98881b(I: Grid) -> Grid:
    x0 = width(I)
    x1 = increment(x0)
    x2 = divide(x1, THREE)
    x3 = decrement(x2)
    x4 = height(I)
    x5 = astuple(x4, x3)
    x6 = crop(I, ORIGIN, x5)
    x7 = add(x3, ONE)
    x8 = tojvec(x7)
    x9 = crop(I, x8, x5)
    x10 = double(x3)
    x11 = add(x10, TWO)
    x12 = tojvec(x11)
    x13 = crop(I, x12, x5)
    x14 = palette(x6)
    x15 = palette(x9)
    x16 = palette(x13)
    x17 = intersection(x14, x15)
    x18 = intersection(x17, x16)
    x19 = first(x18)
    x20 = other(x14, x19)
    x21 = other(x15, x19)
    x22 = other(x16, x19)
    x23 = canvas(x19, x5)
    x24 = ofcolor(x6, x20)
    x25 = ofcolor(x9, x21)
    x26 = ofcolor(x13, x22)
    x27 = fill(x23, x22, x26)
    x28 = fill(x27, x21, x25)
    x29 = fill(x28, x20, x24)
    return x29