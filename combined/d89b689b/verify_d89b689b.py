import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d89b689b(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = sfilter(x0, square)
    x2 = argmax(x1, size)
    x3 = toindices(x2)
    x4 = sizefilter(x1, ONE)
    x5 = apply(initset, x3)
    x6 = lbind(argmin, x5)
    x7 = lbind(rbind, manhattan)
    x8 = compose(x6, x7)
    x9 = fork(recolor, color, x8)
    x10 = mapply(x9, x4)
    x11 = merge(x4)
    x12 = cover(I, x11)
    x13 = paint(x12, x10)
    return x13