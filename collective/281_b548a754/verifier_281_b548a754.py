import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b548a754(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(equality, toindices, box)
    x2 = fork(multiply, height, width)
    x3 = fork(equality, size, x2)
    x4 = compose(flip, x3)
    x5 = fork(both, x1, x4)
    x6 = extract(x0, x5)
    x7 = inbox(x6)
    x8 = backdrop(x7)
    x9 = toobject(x8, I)
    x10 = remove(x9, x0)
    x11 = remove(x6, x10)
    x12 = argmin(x11, size)
    x13 = combine(x12, x6)
    x14 = backdrop(x13)
    x15 = color(x9)
    x16 = fill(I, x15, x14)
    x17 = color(x6)
    x18 = box(x14)
    x19 = fill(x16, x17, x18)
    return x19