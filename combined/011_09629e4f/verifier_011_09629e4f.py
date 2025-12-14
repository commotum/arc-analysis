import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_09629e4f(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = sfilter(x0, hline)
    x2 = sfilter(x0, vline)
    x3 = size(x1)
    x4 = size(x2)
    x5 = merge(x0)
    x6 = color(x5)
    x7 = shape(I)
    x8 = canvas(x6, x7)
    x9 = hconcat(I, x8)
    x10 = objects(x9, F, T, T)
    x11 = argmin(x10, numcolors)
    x12 = normalize(x11)
    x13 = toindices(x12)
    x14 = increment(x3)
    x15 = increment(x14)
    x16 = increment(x4)
    x17 = increment(x16)
    x18 = astuple(x15, x17)
    x19 = lbind(shift, x13)
    x20 = rbind(multiply, x18)
    x21 = chain(x19, x20, last)
    x22 = fork(recolor, first, x21)
    x23 = normalize(x11)
    x24 = mapply(x22, x23)
    x25 = paint(x8, x24)
    return x25