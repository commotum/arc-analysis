import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_75b8110e(I: Grid) -> Grid:
    x0 = tophalf(I)
    x1 = lefthalf(x0)
    x2 = tophalf(I)
    x3 = righthalf(x2)
    x4 = bottomhalf(I)
    x5 = righthalf(x4)
    x6 = bottomhalf(I)
    x7 = lefthalf(x6)
    x8 = palette(x1)
    x9 = palette(x3)
    x10 = intersection(x8, x9)
    x11 = palette(x5)
    x12 = palette(x7)
    x13 = intersection(x11, x12)
    x14 = intersection(x10, x13)
    x15 = first(x14)
    x16 = shape(x1)
    x17 = canvas(x15, x16)
    x18 = matcher(first, x15)
    x19 = compose(flip, x18)
    x20 = rbind(sfilter, x19)
    x21 = compose(x20, asobject)
    x22 = x21(x1)
    x23 = x21(x5)
    x24 = x21(x7)
    x25 = x21(x3)
    x26 = paint(x17, x22)
    x27 = paint(x26, x23)
    x28 = paint(x27, x24)
    x29 = paint(x28, x25)
    return x29