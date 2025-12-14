import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6b9890af(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = fork(equality, toindices, box)
    x2 = sfilter(x0, x1)
    x3 = fork(multiply, height, width)
    x4 = argmax(x2, x3)
    x5 = fgpartition(I)
    x6 = merge(x5)
    x7 = difference(x6, x4)
    x8 = subgrid(x4, I)
    x9 = subgrid(x7, I)
    x10 = height(x8)
    x11 = subtract(x10, TWO)
    x12 = height(x9)
    x13 = divide(x11, x12)
    x14 = width(x8)
    x15 = subtract(x14, TWO)
    x16 = width(x9)
    x17 = divide(x15, x16)
    x18 = hupscale(x9, x17)
    x19 = vupscale(x18, x13)
    x20 = asobject(x19)
    x21 = shift(x20, UNITY)
    x22 = paint(x8, x21)
    return x22