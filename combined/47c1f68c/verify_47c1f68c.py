import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_47c1f68c(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = compress(I)
    x4 = mostcolor(x3)
    x5 = tophalf(I)
    x6 = lefthalf(x5)
    x7 = vmirror(x6)
    x8 = hconcat(x6, x7)
    x9 = hmirror(x8)
    x10 = vconcat(x8, x9)
    x11 = tophalf(I)
    x12 = righthalf(x11)
    x13 = vmirror(x12)
    x14 = hconcat(x13, x12)
    x15 = hmirror(x14)
    x16 = vconcat(x14, x15)
    x17 = bottomhalf(I)
    x18 = lefthalf(x17)
    x19 = vmirror(x18)
    x20 = hconcat(x18, x19)
    x21 = hmirror(x20)
    x22 = vconcat(x21, x20)
    x23 = bottomhalf(I)
    x24 = righthalf(x23)
    x25 = vmirror(x24)
    x26 = hconcat(x25, x24)
    x27 = hmirror(x26)
    x28 = vconcat(x27, x26)
    x29 = astuple(x10, x16)
    x30 = astuple(x22, x28)
    x31 = combine(x29, x30)
    x32 = argmax(x31, numcolors)
    x33 = asindices(x32)
    x34 = ofcolor(x32, x4)
    x35 = difference(x33, x34)
    x36 = fill(x32, x2, x35)
    return x36