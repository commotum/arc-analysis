import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6e19193c(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = rbind(shoot, UNITY)
    x2 = rbind(add, UNITY)
    x3 = chain(x1, x2, lrcorner)
    x4 = fork(recolor, color, x3)
    x5 = rbind(shoot, UP_RIGHT)
    x6 = rbind(add, UP_RIGHT)
    x7 = chain(x5, x6, urcorner)
    x8 = fork(recolor, color, x7)
    x9 = rbind(shoot, NEG_UNITY)
    x10 = rbind(add, NEG_UNITY)
    x11 = chain(x9, x10, ulcorner)
    x12 = fork(recolor, color, x11)
    x13 = rbind(shoot, DOWN_LEFT)
    x14 = rbind(add, DOWN_LEFT)
    x15 = chain(x13, x14, llcorner)
    x16 = fork(recolor, color, x15)
    x17 = fork(remove, lrcorner, toindices)
    x18 = fork(equality, toindices, x17)
    x19 = sfilter(x0, x18)
    x20 = fork(remove, urcorner, toindices)
    x21 = fork(equality, toindices, x20)
    x22 = sfilter(x0, x21)
    x23 = fork(remove, ulcorner, toindices)
    x24 = fork(equality, toindices, x23)
    x25 = sfilter(x0, x24)
    x26 = fork(remove, llcorner, toindices)
    x27 = fork(equality, toindices, x26)
    x28 = sfilter(x0, x27)
    x29 = mapply(x4, x19)
    x30 = mapply(x8, x22)
    x31 = combine(x29, x30)
    x32 = mapply(x12, x25)
    x33 = mapply(x16, x28)
    x34 = combine(x32, x33)
    x35 = combine(x31, x34)
    x36 = paint(I, x35)
    return x36