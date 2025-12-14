import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_444801d8(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = sizefilter(x0, ONE)
    x2 = difference(x0, x1)
    x3 = rbind(toobject, I)
    x4 = chain(leastcolor, x3, delta)
    x5 = rbind(shift, UP)
    x6 = fork(connect, ulcorner, urcorner)
    x7 = compose(x5, x6)
    x8 = rbind(shift, DOWN)
    x9 = fork(connect, llcorner, lrcorner)
    x10 = compose(x8, x9)
    x11 = fork(astuple, x7, x10)
    x12 = lbind(rbind, manhattan)
    x13 = compose(x12, delta)
    x14 = fork(argmin, x11, x13)
    x15 = fork(combine, delta, x14)
    x16 = fork(recolor, x4, x15)
    x17 = mapply(x16, x2)
    x18 = paint(I, x17)
    return x18