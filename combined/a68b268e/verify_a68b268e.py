import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a68b268e(I: Grid) -> Grid:
    x0 = tophalf(I)
    x1 = lefthalf(x0)
    x2 = tophalf(I)
    x3 = righthalf(x2)
    x4 = bottomhalf(I)
    x5 = lefthalf(x4)
    x6 = bottomhalf(I)
    x7 = righthalf(x6)
    x8 = palette(x1)
    x9 = palette(x3)
    x10 = intersection(x8, x9)
    x11 = palette(x5)
    x12 = palette(x7)
    x13 = intersection(x11, x12)
    x14 = intersection(x10, x13)
    x15 = first(x14)
    x16 = shape(I)
    x17 = halve(x16)
    x18 = canvas(x15, x17)
    x19 = matcher(first, x15)
    x20 = compose(flip, x19)
    x21 = rbind(sfilter, x20)
    x22 = compose(x21, asobject)
    x23 = x22(x1)
    x24 = x22(x3)
    x25 = x22(x5)
    x26 = x22(x7)
    x27 = paint(x18, x26)
    x28 = paint(x27, x25)
    x29 = paint(x28, x24)
    x30 = paint(x29, x23)
    return x30