import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6773b310(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = palette(I)
    x7 = remove(x2, x6)
    x8 = lbind(colorcount, I)
    x9 = argmin(x7, x8)
    x10 = other(x7, x9)
    x11 = objects(x5, F, T, T)
    x12 = rbind(colorcount, x9)
    x13 = valmax(x11, x12)
    x14 = rbind(colorcount, x9)
    x15 = matcher(x14, x13)
    x16 = sfilter(x11, x15)
    x17 = apply(ulcorner, x16)
    x18 = first(x11)
    x19 = shape(x18)
    x20 = increment(x19)
    x21 = rbind(divide, x20)
    x22 = apply(x21, x17)
    x23 = sfilter(x0, hline)
    x24 = size(x23)
    x25 = sfilter(x0, vline)
    x26 = size(x25)
    x27 = astuple(x24, x26)
    x28 = increment(x27)
    x29 = canvas(x10, x28)
    x30 = fill(x29, ONE, x22)
    return x30