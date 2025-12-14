import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f2829549(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = sfilter(x0, hline)
    x2 = size(x1)
    x3 = positive(x2)
    x4 = branch(x3, vsplit, hsplit)
    x5 = x4(I, TWO)
    x6 = first(x5)
    x7 = last(x5)
    x8 = palette(x6)
    x9 = palette(x7)
    x10 = intersection(x8, x9)
    x11 = first(x10)
    x12 = shape(x6)
    x13 = canvas(x11, x12)
    x14 = ofcolor(x6, x11)
    x15 = ofcolor(x7, x11)
    x16 = intersection(x14, x15)
    x17 = fill(x13, THREE, x16)
    return x17