import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f1cefba8(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(multiply, height, width)
    x2 = argmax(x0, x1)
    x3 = argmin(x0, x1)
    x4 = remove(x2, x0)
    x5 = other(x4, x3)
    x6 = color(x3)
    x7 = color(x5)
    x8 = toindices(x3)
    x9 = inbox(x5)
    x10 = intersection(x8, x9)
    x11 = fork(combine, hfrontier, vfrontier)
    x12 = mapply(x11, x10)
    x13 = corners(x5)
    x14 = inbox(x5)
    x15 = corners(x14)
    x16 = combine(x13, x15)
    x17 = mapply(x11, x16)
    x18 = difference(x12, x17)
    x19 = toindices(x2)
    x20 = intersection(x18, x19)
    x21 = fill(I, x6, x20)
    x22 = difference(x18, x20)
    x23 = fill(x21, x7, x22)
    x24 = inbox(x5)
    x25 = fill(x23, x7, x24)
    return x25