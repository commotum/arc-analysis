import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ce4f8723(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = sfilter(x0, hline)
    x2 = size(x1)
    x3 = positive(x2)
    x4 = branch(x3, tophalf, lefthalf)
    x5 = branch(x3, bottomhalf, righthalf)
    x6 = x4(I)
    x7 = x5(I)
    x8 = palette(x6)
    x9 = palette(x7)
    x10 = intersection(x8, x9)
    x11 = first(x10)
    x12 = shape(x6)
    x13 = canvas(x11, x12)
    x14 = palette(x6)
    x15 = other(x14, x11)
    x16 = palette(x7)
    x17 = other(x16, x11)
    x18 = ofcolor(x6, x15)
    x19 = ofcolor(x7, x17)
    x20 = combine(x18, x19)
    x21 = fill(x13, THREE, x20)
    return x21