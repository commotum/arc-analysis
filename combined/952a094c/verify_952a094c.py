import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_952a094c(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = inbox(x1)
    x3 = cover(I, x2)
    x4 = ulcorner(x2)
    x5 = index(I, x4)
    x6 = lrcorner(x1)
    x7 = add(UNITY, x6)
    x8 = initset(x7)
    x9 = fill(x3, x5, x8)
    x10 = lrcorner(x2)
    x11 = index(I, x10)
    x12 = ulcorner(x1)
    x13 = add(NEG_UNITY, x12)
    x14 = initset(x13)
    x15 = fill(x9, x11, x14)
    x16 = urcorner(x2)
    x17 = index(I, x16)
    x18 = llcorner(x1)
    x19 = add(DOWN_LEFT, x18)
    x20 = initset(x19)
    x21 = fill(x15, x17, x20)
    x22 = llcorner(x2)
    x23 = index(I, x22)
    x24 = urcorner(x1)
    x25 = add(UP_RIGHT, x24)
    x26 = initset(x25)
    x27 = fill(x21, x23, x26)
    return x27