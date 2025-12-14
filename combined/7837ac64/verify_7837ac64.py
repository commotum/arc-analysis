import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_7837ac64(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = fork(equality, toindices, backdrop)
    x2 = sfilter(x0, x1)
    x3 = fork(multiply, height, width)
    x4 = argmax(x2, x3)
    x5 = color(x4)
    x6 = palette(I)
    x7 = remove(x5, x6)
    x8 = lbind(colorcount, I)
    x9 = argmax(x7, x8)
    x10 = remove(x9, x7)
    x11 = lbind(ofcolor, I)
    x12 = mapply(x11, x10)
    x13 = subgrid(x12, I)
    x14 = objects(x13, T, F, F)
    x15 = colorfilter(x14, x5)
    x16 = initset(x9)
    x17 = insert(x5, x16)
    x18 = lbind(intersection, x17)
    x19 = chain(positive, size, x18)
    x20 = chain(positive, decrement, size)
    x21 = fork(either, x19, x20)
    x22 = rbind(toobject, x13)
    x23 = compose(corners, outbox)
    x24 = chain(palette, x22, x23)
    x25 = rbind(branch, x5)
    x26 = chain(flip, x21, x24)
    x27 = compose(first, x24)
    x28 = fork(x25, x26, x27)
    x29 = apply(uppermost, x15)
    x30 = order(x29, identity)
    x31 = lbind(apply, x28)
    x32 = rbind(order, leftmost)
    x33 = lbind(sfilter, x15)
    x34 = lbind(matcher, uppermost)
    x35 = compose(x33, x34)
    x36 = chain(x31, x32, x35)
    x37 = apply(x36, x30)
    return x37