import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6430c8c4(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = sfilter(x0, hline)
    x2 = size(x1)
    x3 = positive(x2)
    x4 = branch(x3, tophalf, lefthalf)
    x5 = branch(x3, bottomhalf, righthalf)
    x6 = x4(I)
    x7 = x5(I)
    x8 = shape(x6)
    x9 = palette(x6)
    x10 = palette(x7)
    x11 = intersection(x9, x10)
    x12 = first(x11)
    x13 = ofcolor(x6, x12)
    x14 = ofcolor(x7, x12)
    x15 = intersection(x13, x14)
    x16 = canvas(x12, x8)
    x17 = fill(x16, THREE, x15)
    return x17