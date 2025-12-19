import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_0a938d79(I: Grid) -> Grid:
    x0 = portrait(I)
    x1 = branch(x0, dmirror, identity)
    x2 = x1(I)
    x3 = objects(x2, T, F, T)
    x4 = argmin(x3, leftmost)
    x5 = argmax(x3, leftmost)
    x6 = color(x4)
    x7 = color(x5)
    x8 = leftmost(x4)
    x9 = leftmost(x5)
    x10 = subtract(x9, x8)
    x11 = double(x10)
    x12 = multiply(THREE, TEN)
    x13 = interval(x8, x12, x11)
    x14 = interval(x9, x12, x11)
    x15 = compose(vfrontier, tojvec)
    x16 = mapply(x15, x13)
    x17 = mapply(x15, x14)
    x18 = recolor(x6, x16)
    x19 = recolor(x7, x17)
    x20 = combine(x18, x19)
    x21 = paint(x2, x20)
    x22 = x1(x21)
    return x22