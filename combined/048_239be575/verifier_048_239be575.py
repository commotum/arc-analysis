import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_239be575(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = lbind(apply, normalize)
    x2 = lbind(colorfilter, x0)
    x3 = chain(size, x1, x2)
    x4 = matcher(x3, ONE)
    x5 = lbind(colorcount, I)
    x6 = matcher(x5, EIGHT)
    x7 = lbind(colorfilter, x0)
    x8 = compose(size, x7)
    x9 = matcher(x8, TWO)
    x10 = fork(both, x6, x9)
    x11 = fork(both, x10, x4)
    x12 = palette(I)
    x13 = extract(x12, x11)
    x14 = colorfilter(x0, x13)
    x15 = totuple(x14)
    x16 = first(x15)
    x17 = last(x15)
    x18 = palette(I)
    x19 = remove(ZERO, x18)
    x20 = remove(x13, x19)
    x21 = first(x20)
    x22 = colorfilter(x0, x21)
    x23 = rbind(adjacent, x16)
    x24 = rbind(adjacent, x17)
    x25 = fork(both, x23, x24)
    x26 = sfilter(x22, x25)
    x27 = size(x26)
    x28 = positive(x27)
    x29 = branch(x28, x21, ZERO)
    x30 = canvas(x29, UNITY)
    return x30