import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3befdf3e(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = rbind(compose, last)
    x2 = lbind(rbind, contained)
    x3 = chain(x1, x2, box)
    x4 = fork(sfilter, identity, x3)
    x5 = compose(color, x4)
    x6 = fork(other, palette, x5)
    x7 = chain(decrement, decrement, height)
    x8 = chain(decrement, decrement, width)
    x9 = compose(toivec, x7)
    x10 = fork(shift, toindices, x9)
    x11 = chain(toivec, invert, x7)
    x12 = fork(shift, toindices, x11)
    x13 = compose(tojvec, x8)
    x14 = fork(shift, toindices, x13)
    x15 = chain(tojvec, invert, x8)
    x16 = fork(shift, toindices, x15)
    x17 = fork(combine, x10, x12)
    x18 = fork(combine, x14, x16)
    x19 = fork(combine, x17, x18)
    x20 = fork(combine, backdrop, x19)
    x21 = fork(difference, x20, box)
    x22 = fork(recolor, x5, x21)
    x23 = fork(recolor, x6, box)
    x24 = fork(combine, x22, x23)
    x25 = mapply(x24, x0)
    x26 = paint(I, x25)
    return x26