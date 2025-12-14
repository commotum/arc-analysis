import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e3497940(I: Grid) -> Grid:
    x0 = lefthalf(I)
    x1 = righthalf(I)
    x2 = vmirror(x1)
    x3 = width(I)
    x4 = hsplit(I, x3)
    x5 = first(x4)
    x6 = mostcolor(x5)
    x7 = objects(x2, T, F, F)
    x8 = matcher(color, x6)
    x9 = compose(flip, x8)
    x10 = sfilter(x7, x9)
    x11 = merge(x10)
    x12 = paint(x0, x11)
    return x12