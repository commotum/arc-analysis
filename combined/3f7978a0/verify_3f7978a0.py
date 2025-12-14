import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3f7978a0(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = objects(I, T, F, F)
    x2 = compose(double, height)
    x3 = fork(equality, x2, size)
    x4 = compose(double, width)
    x5 = fork(equality, x4, size)
    x6 = fork(either, x3, x5)
    x7 = rbind(equality, TWO)
    x8 = lbind(colorfilter, x1)
    x9 = rbind(sfilter, vline)
    x10 = rbind(sfilter, hline)
    x11 = chain(x9, x8, color)
    x12 = chain(x7, size, x11)
    x13 = chain(x10, x8, color)
    x14 = chain(x7, size, x13)
    x15 = fork(either, x12, x14)
    x16 = fork(both, x6, x15)
    x17 = extract(x0, x16)
    x18 = color(x17)
    x19 = colorfilter(x1, x18)
    x20 = first(x19)
    x21 = vline(x20)
    x22 = ulcorner(x17)
    x23 = lrcorner(x17)
    x24 = branch(x21, UP, LEFT)
    x25 = add(x22, x24)
    x26 = branch(x21, DOWN, RIGHT)
    x27 = add(x23, x26)
    x28 = initset(x27)
    x29 = insert(x25, x28)
    x30 = subgrid(x29, I)
    return x30