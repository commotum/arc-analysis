import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1b2d62fb(I: Grid) -> Grid:
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
    x14 = shape(x12)
    x15 = canvas(ZERO, x14)
    x16 = ofcolor(x12, ZERO)
    x17 = ofcolor(x13, ZERO)
    x18 = intersection(x16, x17)
    x19 = fill(x15, EIGHT, x18)
    return x19