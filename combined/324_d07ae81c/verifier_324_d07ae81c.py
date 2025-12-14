import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d07ae81c(I: Grid) -> Grid:
    x0 = lbind(ofcolor, I)
    x1 = lbind(mapply, neighbors)
    x2 = compose(x1, x0)
    x3 = fork(intersection, x0, x2)
    x4 = compose(size, x3)
    x5 = palette(I)
    x6 = matcher(x4, ZERO)
    x7 = sfilter(x5, x6)
    x8 = totuple(x7)
    x9 = first(x8)
    x10 = last(x8)
    x11 = ofcolor(I, x9)
    x12 = mapply(neighbors, x11)
    x13 = toobject(x12, I)
    x14 = mostcolor(x13)
    x15 = ofcolor(I, x10)
    x16 = mapply(neighbors, x15)
    x17 = toobject(x16, I)
    x18 = mostcolor(x17)
    x19 = rbind(shoot, UNITY)
    x20 = rbind(shoot, NEG_UNITY)
    x21 = fork(combine, x19, x20)
    x22 = rbind(shoot, UP_RIGHT)
    x23 = rbind(shoot, DOWN_LEFT)
    x24 = fork(combine, x22, x23)
    x25 = fork(combine, x21, x24)
    x26 = ofcolor(I, x10)
    x27 = ofcolor(I, x9)
    x28 = combine(x26, x27)
    x29 = mapply(x25, x28)
    x30 = ofcolor(I, x14)
    x31 = intersection(x30, x29)
    x32 = ofcolor(I, x18)
    x33 = intersection(x32, x29)
    x34 = fill(I, x9, x31)
    x35 = fill(x34, x10, x33)
    return x35