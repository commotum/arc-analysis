import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e8dc4411(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = argmax(x0, size)
    x2 = other(x0, x1)
    x3 = ineighbors(ORIGIN)
    x4 = height(x1)
    x5 = increment(x4)
    x6 = interval(ZERO, x5, ONE)
    x7 = lbind(intersection, x1)
    x8 = chain(positive, size, x7)
    x9 = lbind(shift, x1)
    x10 = rbind(multiply, UNITY)
    x11 = chain(x8, x9, x10)
    x12 = sfilter(x6, x11)
    x13 = maximum(x12)
    x14 = increment(x13)
    x15 = toindices(x2)
    x16 = lbind(intersection, x15)
    x17 = lbind(shift, x1)
    x18 = rbind(multiply, x14)
    x19 = chain(toindices, x17, x18)
    x20 = chain(size, x16, x19)
    x21 = argmax(x3, x20)
    x22 = shape(I)
    x23 = maximum(x22)
    x24 = increment(x23)
    x25 = interval(ONE, x24, ONE)
    x26 = lbind(shift, x1)
    x27 = multiply(x14, x21)
    x28 = lbind(multiply, x27)
    x29 = pair(x25, x25)
    x30 = apply(x28, x29)
    x31 = mapply(x26, x30)
    x32 = color(x2)
    x33 = recolor(x32, x31)
    x34 = paint(I, x33)
    return x34