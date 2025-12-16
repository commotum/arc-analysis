import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d9fac9be(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = palette(I)
    x2 = remove(x0, x1)
    x3 = totuple(x2)
    x4 = first(x3)
    x5 = last(x3)
    x6 = neighbors(UNITY)
    x7 = initset(UNITY)
    x8 = recolor(x4, x6)
    x9 = recolor(x5, x7)
    x10 = combine(x8, x9)
    x11 = occurrences(I, x10)
    x12 = size(x11)
    x13 = positive(x12)
    x14 = branch(x13, x5, x4)
    x15 = canvas(x14, UNITY)
    return x15