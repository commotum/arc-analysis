import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_137eaa0f(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = palette(x1)
    x3 = objects(I, T, F, T)
    x4 = totuple(x3)
    x5 = apply(color, x4)
    x6 = lbind(sfilter, x5)
    x7 = lbind(matcher, identity)
    x8 = chain(size, x6, x7)
    x9 = valmax(x2, x8)
    x10 = matcher(x8, x9)
    x11 = sfilter(x2, x10)
    x12 = lbind(colorcount, I)
    x13 = argmin(x11, x12)
    x14 = ofcolor(I, x13)
    x15 = recolor(x13, x14)
    x16 = apply(initset, x15)
    x17 = remove(x15, x0)
    x18 = lbind(argmin, x16)
    x19 = lbind(rbind, manhattan)
    x20 = compose(x18, x19)
    x21 = fork(combine, identity, x20)
    x22 = apply(x21, x17)
    x23 = matcher(first, x13)
    x24 = rbind(sfilter, x23)
    x25 = chain(invert, ulcorner, x24)
    x26 = fork(shift, identity, x25)
    x27 = mapply(x26, x22)
    x28 = normalize(x27)
    x29 = shape(x28)
    x30 = canvas(ZERO, x29)
    x31 = paint(x30, x28)
    return x31