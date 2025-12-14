import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1f642eb9(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = fork(multiply, height, width)
    x2 = fork(equality, size, x1)
    x3 = sfilter(x0, x2)
    x4 = argmax(x3, size)
    x5 = outbox(x4)
    x6 = corners(x5)
    x7 = toobject(x6, I)
    x8 = color(x7)
    x9 = asindices(I)
    x10 = ofcolor(I, x8)
    x11 = toindices(x4)
    x12 = combine(x10, x11)
    x13 = difference(x9, x12)
    x14 = toobject(x13, I)
    x15 = apply(initset, x14)
    x16 = rbind(gravitate, x4)
    x17 = compose(crement, x16)
    x18 = fork(shift, identity, x17)
    x19 = mapply(x18, x15)
    x20 = paint(I, x19)
    return x20