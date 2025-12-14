import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_8403a5d5(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = argmin(x0, size)
    x2 = color(x1)
    x3 = leftmost(x1)
    x4 = width(I)
    x5 = interval(x3, x4, TWO)
    x6 = apply(tojvec, x5)
    x7 = mapply(vfrontier, x6)
    x8 = fill(I, x2, x7)
    x9 = increment(x3)
    x10 = width(I)
    x11 = interval(x9, x10, FOUR)
    x12 = add(x3, THREE)
    x13 = width(I)
    x14 = interval(x12, x13, FOUR)
    x15 = apply(tojvec, x11)
    x16 = height(I)
    x17 = decrement(x16)
    x18 = lbind(astuple, x17)
    x19 = apply(x18, x14)
    x20 = combine(x15, x19)
    x21 = fill(x8, FIVE, x20)
    return x21