import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_045e512c(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = argmax(x0, size)
    x2 = height(x1)
    x3 = width(x1)
    x4 = neighbors(ORIGIN)
    x5 = toindices(x1)
    x6 = lbind(shift, x5)
    x7 = height(I)
    x8 = divide(x7, x2)
    x9 = width(I)
    x10 = divide(x9, x3)
    x11 = astuple(x8, x10)
    x12 = maximum(x11)
    x13 = increment(x12)
    x14 = interval(ONE, x13, ONE)
    x15 = astuple(x2, x3)
    x16 = lbind(multiply, x15)
    x17 = compose(crement, x16)
    x18 = lbind(mapply, x6)
    x19 = rbind(apply, x14)
    x20 = lbind(rbind, multiply)
    x21 = compose(x20, x17)
    x22 = chain(x18, x19, x21)
    x23 = rbind(toobject, I)
    x24 = compose(x6, x17)
    x25 = chain(palette, x23, x24)
    x26 = mostcolor(I)
    x27 = rbind(equality, x26)
    x28 = rbind(argmin, x27)
    x29 = compose(x28, x25)
    x30 = fork(recolor, x29, x22)
    x31 = mapply(x30, x4)
    x32 = paint(I, x31)
    return x32