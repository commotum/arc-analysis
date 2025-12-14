import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a48eeaf7(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = fork(equality, toindices, backdrop)
    x2 = sfilter(x0, x1)
    x3 = argmax(x2, size)
    x4 = other(x0, x3)
    x5 = color(x4)
    x6 = toindices(x4)
    x7 = outbox(x3)
    x8 = lbind(argmin, x7)
    x9 = lbind(lbind, manhattan)
    x10 = rbind(compose, initset)
    x11 = chain(x8, x10, x9)
    x12 = compose(x11, initset)
    x13 = apply(x12, x6)
    x14 = cover(I, x4)
    x15 = fill(x14, x5, x13)
    return x15