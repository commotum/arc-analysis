import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_fafffa47(I: Grid) -> Grid:
    x0 = tophalf(I)
    x1 = numcolors(x0)
    x2 = equality(x1, TWO)
    x3 = bottomhalf(I)
    x4 = numcolors(x3)
    x5 = equality(x4, TWO)
    x6 = both(x2, x5)
    x7 = lefthalf(I)
    x8 = numcolors(x7)
    x9 = equality(x8, TWO)
    x10 = righthalf(I)
    x11 = numcolors(x10)
    x12 = equality(x11, TWO)
    x13 = both(x9, x12)
    x14 = flip(x13)
    x15 = both(x6, x14)
    x16 = branch(x15, vsplit, hsplit)
    x17 = x16(I, TWO)
    x18 = first(x17)
    x19 = last(x17)
    x20 = palette(x18)
    x21 = palette(x19)
    x22 = intersection(x20, x21)
    x23 = first(x22)
    x24 = shape(x18)
    x25 = canvas(x23, x24)
    x26 = ofcolor(x18, x23)
    x27 = ofcolor(x19, x23)
    x28 = intersection(x26, x27)
    x29 = fill(x25, TWO, x28)
    return x29