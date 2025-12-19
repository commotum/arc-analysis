import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_469497ad(I: Grid) -> Grid:
    x0 = numcolors(I)
    x1 = decrement(x0)
    x2 = upscale(I, x1)
    x3 = rbind(toobject, I)
    x4 = lbind(ofcolor, I)
    x5 = compose(outbox, x4)
    x6 = chain(numcolors, x3, x5)
    x7 = matcher(x6, ONE)
    x8 = palette(I)
    x9 = sfilter(x8, x7)
    x10 = fork(multiply, height, width)
    x11 = lbind(ofcolor, I)
    x12 = compose(x10, x11)
    x13 = argmin(x9, x12)
    x14 = ofcolor(x2, x13)
    x15 = outbox(x14)
    x16 = toobject(x15, x2)
    x17 = mostcolor(x16)
    x18 = ulcorner(x14)
    x19 = shoot(x18, NEG_UNITY)
    x20 = lrcorner(x14)
    x21 = shoot(x20, UNITY)
    x22 = urcorner(x14)
    x23 = shoot(x22, UP_RIGHT)
    x24 = llcorner(x14)
    x25 = shoot(x24, DOWN_LEFT)
    x26 = combine(x19, x21)
    x27 = combine(x23, x25)
    x28 = combine(x26, x27)
    x29 = ofcolor(x2, x17)
    x30 = intersection(x28, x29)
    x31 = fill(x2, TWO, x30)
    return x31