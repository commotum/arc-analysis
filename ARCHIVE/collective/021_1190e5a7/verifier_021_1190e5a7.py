import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1190e5a7(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = corners(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = palette(I)
    x5 = rbind(equality, x3)
    x6 = argmin(x4, x5)
    x7 = asindices(I)
    x8 = ofcolor(I, x3)
    x9 = difference(x7, x8)
    x10 = fill(I, x6, x9)
    x11 = frontiers(x10)
    x12 = sfilter(x11, vline)
    x13 = difference(x11, x12)
    x14 = astuple(x13, x12)
    x15 = apply(size, x14)
    x16 = increment(x15)
    x17 = canvas(x3, x16)
    return x17