import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d43fd935(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = asobject(I)
    x2 = matcher(first, x0)
    x3 = compose(flip, x2)
    x4 = sfilter(x1, x3)
    x5 = partition(I)
    x6 = fork(multiply, height, width)
    x7 = fork(equality, size, x6)
    x8 = sfilter(x5, x7)
    x9 = argmax(x8, size)
    x10 = difference(x4, x9)
    x11 = apply(initset, x10)
    x12 = rbind(hmatching, x9)
    x13 = rbind(vmatching, x9)
    x14 = fork(either, x12, x13)
    x15 = sfilter(x11, x14)
    x16 = rbind(gravitate, x9)
    x17 = fork(add, center, x16)
    x18 = fork(connect, center, x17)
    x19 = fork(recolor, color, x18)
    x20 = mapply(x19, x15)
    x21 = paint(I, x20)
    return x21