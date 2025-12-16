import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_23581191(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = palette(I)
    x2 = remove(x0, x1)
    x3 = totuple(x2)
    x4 = fork(combine, hfrontier, vfrontier)
    x5 = lbind(mapply, x4)
    x6 = lbind(ofcolor, I)
    x7 = compose(x5, x6)
    x8 = first(x3)
    x9 = last(x3)
    x10 = x7(x8)
    x11 = x7(x9)
    x12 = ofcolor(I, x0)
    x13 = intersection(x12, x10)
    x14 = intersection(x12, x11)
    x15 = intersection(x10, x11)
    x16 = intersection(x12, x15)
    x17 = fill(I, x8, x13)
    x18 = fill(x17, x9, x14)
    x19 = fill(x18, TWO, x16)
    return x19