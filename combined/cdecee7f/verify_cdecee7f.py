import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_cdecee7f(I: Grid) -> Grid:
    x0 = asobject(I)
    x1 = mostcolor(I)
    x2 = matcher(first, x1)
    x3 = compose(flip, x2)
    x4 = sfilter(x0, x3)
    x5 = apply(initset, x4)
    x6 = astuple(ONE, THREE)
    x7 = size(x5)
    x8 = order(x5, leftmost)
    x9 = apply(color, x8)
    x10 = rbind(canvas, UNITY)
    x11 = apply(x10, x9)
    x12 = merge(x11)
    x13 = dmirror(x12)
    x14 = subtract(NINE, x7)
    x15 = astuple(ONE, x14)
    x16 = mostcolor(I)
    x17 = canvas(x16, x15)
    x18 = hconcat(x13, x17)
    x19 = hsplit(x18, THREE)
    x20 = merge(x19)
    x21 = crop(x20, ORIGIN, x6)
    x22 = crop(x20, DOWN, x6)
    x23 = crop(x20, TWO_BY_ZERO, x6)
    x24 = vmirror(x22)
    x25 = vconcat(x21, x24)
    x26 = vconcat(x25, x23)
    return x26