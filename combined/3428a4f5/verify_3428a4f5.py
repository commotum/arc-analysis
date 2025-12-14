import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3428a4f5(I: Grid) -> Grid:
    x0 = width(I)
    x1 = halve(x0)
    x2 = tojvec(x1)
    x3 = height(I)
    x4 = decrement(x3)
    x5 = astuple(x4, x1)
    x6 = connect(x2, x5)
    x7 = toobject(x6, I)
    x8 = numcolors(x7)
    x9 = equality(x8, ONE)
    x10 = branch(x9, lefthalf, tophalf)
    x11 = branch(x9, righthalf, bottomhalf)
    x12 = x10(I)
    x13 = x11(I)
    x14 = palette(x12)
    x15 = other(x14, ZERO)
    x16 = palette(x13)
    x17 = other(x16, ZERO)
    x18 = shape(x12)
    x19 = canvas(ZERO, x18)
    x20 = ofcolor(x12, x15)
    x21 = ofcolor(x13, x17)
    x22 = combine(x20, x21)
    x23 = intersection(x20, x21)
    x24 = difference(x22, x23)
    x25 = fill(x19, THREE, x24)
    return x25