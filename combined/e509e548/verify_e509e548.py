import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e509e548(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = fork(add, height, width)
    x2 = compose(decrement, x1)
    x3 = fork(equality, size, x2)
    x4 = fork(difference, toindices, box)
    x5 = compose(size, x4)
    x6 = matcher(x5, ZERO)
    x7 = sfilter(x0, x3)
    x8 = difference(x0, x7)
    x9 = sfilter(x8, x6)
    x10 = merge(x0)
    x11 = fill(I, TWO, x10)
    x12 = merge(x7)
    x13 = fill(x11, ONE, x12)
    x14 = merge(x9)
    x15 = fill(x13, SIX, x14)
    return x15