import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_025d127b(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = objects(I, T, T, T)
    x2 = rbind(objects, F)
    x3 = rbind(x2, F)
    x4 = rbind(x3, T)
    x5 = lbind(canvas, x0)
    x6 = compose(x5, shape)
    x7 = fork(paint, x6, normalize)
    x8 = compose(x4, x7)
    x9 = fork(colorfilter, x8, color)
    x10 = rbind(shift, RIGHT)
    x11 = rbind(argmax, rightmost)
    x12 = compose(x11, x9)
    x13 = fork(remove, x12, x9)
    x14 = chain(x10, merge, x13)
    x15 = rbind(argmax, rightmost)
    x16 = compose(x15, x9)
    x17 = fork(combine, x16, x14)
    x18 = fork(shift, x17, ulcorner)
    x19 = merge(x1)
    x20 = fill(I, x0, x19)
    x21 = mapply(x18, x1)
    x22 = paint(x20, x21)
    return x22