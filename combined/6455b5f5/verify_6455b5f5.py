import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6455b5f5(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = asindices(I)
    x2 = corners(x1)
    x3 = toobject(x2, I)
    x4 = mostcolor(x3)
    x5 = colorfilter(x0, x4)
    x6 = valmax(x5, size)
    x7 = valmin(x5, size)
    x8 = sizefilter(x5, x6)
    x9 = sizefilter(x5, x7)
    x10 = merge(x8)
    x11 = fill(I, ONE, x10)
    x12 = merge(x9)
    x13 = fill(x11, EIGHT, x12)
    return x13