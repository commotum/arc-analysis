import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_77fdfe62(I: Grid) -> Grid:
    x0 = trim(I)
    x1 = trim(x0)
    x2 = tophalf(x1)
    x3 = lefthalf(x2)
    x4 = tophalf(x1)
    x5 = righthalf(x4)
    x6 = bottomhalf(x1)
    x7 = lefthalf(x6)
    x8 = bottomhalf(x1)
    x9 = righthalf(x8)
    x10 = index(I, ORIGIN)
    x11 = width(I)
    x12 = decrement(x11)
    x13 = tojvec(x12)
    x14 = index(I, x13)
    x15 = height(I)
    x16 = decrement(x15)
    x17 = toivec(x16)
    x18 = index(I, x17)
    x19 = shape(I)
    x20 = decrement(x19)
    x21 = index(I, x20)
    x22 = compress(I)
    x23 = asindices(x22)
    x24 = box(x23)
    x25 = corners(x23)
    x26 = difference(x24, x25)
    x27 = toobject(x26, x22)
    x28 = color(x27)
    x29 = palette(x1)
    x30 = other(x29, x28)
    x31 = replace(x3, x30, x10)
    x32 = replace(x5, x30, x14)
    x33 = replace(x7, x30, x18)
    x34 = replace(x9, x30, x21)
    x35 = hconcat(x31, x32)
    x36 = hconcat(x33, x34)
    x37 = vconcat(x35, x36)
    return x37