import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a78176bb(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = objects(I, T, T, F)
    x2 = fork(connect, ulcorner, lrcorner)
    x3 = fork(equality, toindices, x2)
    x4 = sfilter(x1, x3)
    x5 = size(x4)
    x6 = positive(x5)
    x7 = branch(x6, identity, hmirror)
    x8 = x7(I)
    x9 = objects(x8, T, F, T)
    x10 = compose(flip, x3)
    x11 = sfilter(x9, x10)
    x12 = rbind(shoot, UNITY)
    x13 = rbind(shoot, NEG_UNITY)
    x14 = fork(combine, x12, x13)
    x15 = rbind(branch, llcorner)
    x16 = rbind(x15, urcorner)
    x17 = rbind(branch, DOWN_LEFT)
    x18 = rbind(x17, UP_RIGHT)
    x19 = rbind(branch, RIGHT)
    x20 = rbind(x19, DOWN)
    x21 = fork(contained, urcorner, toindices)
    x22 = lbind(index, x8)
    x23 = compose(x20, x21)
    x24 = fork(add, ulcorner, x23)
    x25 = compose(x22, x24)
    x26 = chain(initset, x16, x21)
    x27 = fork(rapply, x26, identity)
    x28 = compose(first, x27)
    x29 = compose(x18, x21)
    x30 = fork(add, x28, x29)
    x31 = compose(x14, x30)
    x32 = fork(recolor, x25, x31)
    x33 = mapply(x32, x11)
    x34 = merge(x11)
    x35 = cover(x8, x34)
    x36 = paint(x35, x33)
    x37 = x7(x36)
    return x37