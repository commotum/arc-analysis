import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_54d82841(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = mapply(delta, x0)
    x2 = first(x0)
    x3 = toindices(x2)
    x4 = rbind(contained, x3)
    x5 = portrait(x2)
    x6 = apply(first, x1)
    x7 = apply(last, x1)
    x8 = branch(x5, x6, x7)
    x9 = branch(x5, RIGHT, DOWN)
    x10 = delta(x2)
    x11 = center(x10)
    x12 = add(x11, x9)
    x13 = x4(x12)
    x14 = branch(x5, width, height)
    x15 = branch(x5, rbind, lbind)
    x16 = x14(I)
    x17 = decrement(x16)
    x18 = x15(astuple, x17)
    x19 = branch(x5, toivec, tojvec)
    x20 = branch(x13, x19, x18)
    x21 = apply(x20, x8)
    x22 = fill(I, FOUR, x21)
    return x22