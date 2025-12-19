import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ef135b50(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = fork(multiply, height, width)
    x2 = fork(equality, size, x1)
    x3 = compose(flip, x2)
    x4 = sfilter(x0, x3)
    x5 = argmax(x4, x1)
    x6 = color(x5)
    x7 = ofcolor(I, x6)
    x8 = asindices(I)
    x9 = difference(x8, x7)
    x10 = fill(I, NEG_ONE, x9)
    x11 = lbind(recolor, NEG_ONE)
    x12 = rbind(ofcolor, NEG_ONE)
    x13 = chain(x11, backdrop, x12)
    x14 = fork(paint, identity, x13)
    x15 = height(x10)
    x16 = vsplit(x10, x15)
    x17 = mapply(x14, x16)
    x18 = ofcolor(x17, NEG_ONE)
    x19 = asindices(I)
    x20 = box(x19)
    x21 = difference(x18, x20)
    x22 = intersection(x21, x7)
    x23 = fill(I, NINE, x22)
    return x23