import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e48d4e1a(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = asobject(I)
    x4 = difference(x3, x1)
    x5 = leastcolor(x4)
    x6 = colorcount(I, x5)
    x7 = mostcolor(x4)
    x8 = ofcolor(I, x5)
    x9 = toindices(x1)
    x10 = combine(x9, x8)
    x11 = fill(I, x7, x10)
    x12 = argmax(x0, width)
    x13 = uppermost(x12)
    x14 = argmax(x0, height)
    x15 = leftmost(x14)
    x16 = astuple(x13, x15)
    x17 = initset(x16)
    x18 = position(x8, x17)
    x19 = multiply(x18, x6)
    x20 = add(x16, x19)
    x21 = hfrontier(x20)
    x22 = vfrontier(x20)
    x23 = combine(x21, x22)
    x24 = fill(x11, x2, x23)
    return x24