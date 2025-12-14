import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_fcb5c309(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = lbind(contained, F)
    x2 = compose(flip, x1)
    x3 = fork(equality, toindices, box)
    x4 = lbind(apply, x3)
    x5 = lbind(colorfilter, x0)
    x6 = chain(x2, x4, x5)
    x7 = rbind(greater, TWO)
    x8 = compose(minimum, shape)
    x9 = lbind(apply, x8)
    x10 = chain(x7, minimum, x9)
    x11 = lbind(colorfilter, x0)
    x12 = compose(x10, x11)
    x13 = fork(both, x6, x12)
    x14 = palette(I)
    x15 = extract(x14, x13)
    x16 = palette(I)
    x17 = remove(x15, x16)
    x18 = lbind(colorcount, I)
    x19 = argmin(x17, x18)
    x20 = rbind(colorcount, x19)
    x21 = rbind(toobject, I)
    x22 = chain(x20, x21, backdrop)
    x23 = colorfilter(x0, x15)
    x24 = argmax(x23, x22)
    x25 = subgrid(x24, I)
    x26 = replace(x25, x15, x19)
    return x26