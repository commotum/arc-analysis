import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_9aec4887(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = fork(multiply, height, width)
    x2 = fork(equality, size, x1)
    x3 = sfilter(x0, x2)
    x4 = mostcolor(I)
    x5 = colorfilter(x3, x4)
    x6 = argmax(x5, size)
    x7 = outbox(x6)
    x8 = backdrop(x7)
    x9 = subgrid(x8, I)
    x10 = cover(I, x8)
    x11 = fgpartition(x10)
    x12 = merge(x11)
    x13 = normalize(x12)
    x14 = shift(x13, UNITY)
    x15 = paint(x9, x14)
    x16 = toindices(x14)
    x17 = fgpartition(x9)
    x18 = rbind(remove, x17)
    x19 = lbind(lbind, manhattan)
    x20 = compose(x19, initset)
    x21 = lbind(fork, greater)
    x22 = lbind(sfilter, x16)
    x23 = rbind(compose, x20)
    x24 = lbind(lbind, valmin)
    x25 = chain(x23, x24, x18)
    x26 = rbind(compose, initset)
    x27 = lbind(rbind, manhattan)
    x28 = compose(x26, x27)
    x29 = fork(x21, x25, x28)
    x30 = compose(x22, x29)
    x31 = fork(recolor, color, x30)
    x32 = mapply(x31, x17)
    x33 = paint(x15, x32)
    return x33