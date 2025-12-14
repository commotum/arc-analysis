import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5521c0d9(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = box(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = rbind(objects, T)
    x5 = rbind(x4, F)
    x6 = rbind(x5, T)
    x7 = lbind(canvas, x3)
    x8 = compose(x7, shape)
    x9 = fork(hconcat, identity, x8)
    x10 = compose(x6, x9)
    x11 = lbind(apply, uppermost)
    x12 = chain(maximum, x11, x10)
    x13 = matcher(x12, ZERO)
    x14 = astuple(identity, dmirror)
    x15 = astuple(cmirror, hmirror)
    x16 = combine(x14, x15)
    x17 = rbind(rapply, I)
    x18 = chain(first, x17, initset)
    x19 = compose(x13, x18)
    x20 = extract(x16, x19)
    x21 = x20(I)
    x22 = shape(x21)
    x23 = canvas(x3, x22)
    x24 = hconcat(x21, x23)
    x25 = objects(x24, T, F, T)
    x26 = compose(toivec, height)
    x27 = fork(shift, identity, x26)
    x28 = mapply(x27, x25)
    x29 = mostcolor(I)
    x30 = merge(x25)
    x31 = fill(x21, x29, x30)
    x32 = paint(x31, x28)
    x33 = x20(x32)
    return x33