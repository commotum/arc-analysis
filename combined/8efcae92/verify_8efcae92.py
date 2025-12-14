import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_8efcae92(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = fork(multiply, height, width)
    x2 = argmax(x0, x1)
    x3 = remove(x2, x0)
    x4 = lbind(chain, positive)
    x5 = lbind(x4, size)
    x6 = rbind(compose, backdrop)
    x7 = lbind(lbind, intersection)
    x8 = chain(x5, x6, x7)
    x9 = chain(x8, backdrop, outbox)
    x10 = lbind(sfilter, x3)
    x11 = compose(x10, x9)
    x12 = chain(positive, size, x11)
    x13 = sfilter(x3, x12)
    x14 = compose(merge, x11)
    x15 = apply(x14, x13)
    x16 = rbind(subgrid, I)
    x17 = apply(x16, x15)
    x18 = merge(x15)
    x19 = palette(x18)
    x20 = lbind(colorcount, x18)
    x21 = argmin(x19, x20)
    x22 = rbind(colorcount, x21)
    x23 = argmax(x17, x22)
    return x23