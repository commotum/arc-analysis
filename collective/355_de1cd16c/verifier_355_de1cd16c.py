import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_de1cd16c(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = totuple(x0)
    x2 = apply(color, x1)
    x3 = size(x2)
    x4 = dedupe(x2)
    x5 = size(x4)
    x6 = equality(x3, x5)
    x7 = compose(leastcolor, merge)
    x8 = lbind(apply, color)
    x9 = chain(mostcommon, x8, totuple)
    x10 = branch(x6, x7, x9)
    x11 = x10(x0)
    x12 = objects(I, T, F, F)
    x13 = colorfilter(x12, x11)
    x14 = difference(x12, x13)
    x15 = rbind(subgrid, I)
    x16 = apply(x15, x14)
    x17 = rbind(colorcount, x11)
    x18 = argmax(x16, x17)
    x19 = mostcolor(x18)
    x20 = canvas(x19, UNITY)
    return x20