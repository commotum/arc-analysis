import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_7c008303(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = objects(x5, F, F, T)
    x7 = argmin(x6, size)
    x8 = argmax(x6, size)
    x9 = remove(x8, x6)
    x10 = remove(x7, x9)
    x11 = merge(x10)
    x12 = color(x11)
    x13 = subgrid(x8, I)
    x14 = subgrid(x7, I)
    x15 = width(x8)
    x16 = halve(x15)
    x17 = hupscale(x14, x16)
    x18 = height(x8)
    x19 = halve(x18)
    x20 = vupscale(x17, x19)
    x21 = asobject(x20)
    x22 = asindices(x13)
    x23 = ofcolor(x13, x12)
    x24 = difference(x22, x23)
    x25 = rbind(contained, x24)
    x26 = compose(x25, last)
    x27 = sfilter(x21, x26)
    x28 = paint(x13, x27)
    return x28