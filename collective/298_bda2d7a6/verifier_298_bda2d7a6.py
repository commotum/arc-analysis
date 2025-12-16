import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_bda2d7a6(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = compose(maximum, shape)
    x2 = order(x0, x1)
    x3 = first(x2)
    x4 = last(x2)
    x5 = color(x3)
    x6 = color(x4)
    x7 = equality(x5, x6)
    x8 = combine(x3, x4)
    x9 = repeat(x8, ONE)
    x10 = remove(x3, x2)
    x11 = remove(x4, x10)
    x12 = combine(x9, x11)
    x13 = branch(x7, x12, x2)
    x14 = apply(color, x13)
    x15 = last(x13)
    x16 = remove(x15, x13)
    x17 = repeat(x15, ONE)
    x18 = combine(x17, x16)
    x19 = mpapply(recolor, x14, x18)
    x20 = paint(I, x19)
    return x20