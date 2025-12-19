import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5c2c9af4(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = height(x1)
    x3 = halve(x2)
    x4 = width(x1)
    x5 = halve(x4)
    x6 = ulcorner(x1)
    x7 = lrcorner(x1)
    x8 = shape(I)
    x9 = maximum(x8)
    x10 = multiply(THREE, TEN)
    x11 = interval(ZERO, x10, ONE)
    x12 = rbind(multiply, x3)
    x13 = apply(x12, x11)
    x14 = rbind(multiply, x5)
    x15 = apply(x14, x11)
    x16 = pair(x13, x15)
    x17 = rbind(add, x6)
    x18 = apply(invert, x16)
    x19 = apply(x17, x18)
    x20 = rbind(add, x7)
    x21 = apply(x20, x16)
    x22 = pair(x19, x21)
    x23 = mapply(box, x22)
    x24 = fill(I, x0, x23)
    return x24