import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_c8cbb738(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = valmax(x0, height)
    x2 = valmax(x0, width)
    x3 = astuple(x1, x2)
    x4 = mostcolor(I)
    x5 = canvas(x4, x3)
    x6 = asindices(x5)
    x7 = apply(normalize, x0)
    x8 = box(x6)
    x9 = maximum(x3)
    x10 = double(x9)
    x11 = asindices(x5)
    x12 = center(x11)
    x13 = initset(x12)
    x14 = lbind(manhattan, x13)
    x15 = lbind(multiply, x10)
    x16 = lbind(intersection, x8)
    x17 = chain(x15, size, x16)
    x18 = lbind(fork, subtract)
    x19 = lbind(chain, x17)
    x20 = lbind(x19, toindices)
    x21 = lbind(lbind, shift)
    x22 = compose(x20, x21)
    x23 = lbind(chain, x14)
    x24 = compose(initset, center)
    x25 = lbind(x23, x24)
    x26 = lbind(lbind, shift)
    x27 = compose(x25, x26)
    x28 = lbind(argmax, x6)
    x29 = fork(x18, x22, x27)
    x30 = compose(x28, x29)
    x31 = fork(shift, identity, x30)
    x32 = mapply(x31, x7)
    x33 = paint(x5, x32)
    return x33