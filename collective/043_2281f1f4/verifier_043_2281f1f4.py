import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_2281f1f4(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = apply(first, x1)
    x3 = apply(last, x1)
    x4 = product(x2, x3)
    x5 = difference(x4, x1)
    x6 = fill(I, TWO, x5)
    x7 = lbind(fork, either)
    x8 = lbind(matcher, first)
    x9 = compose(x8, first)
    x10 = lbind(matcher, last)
    x11 = compose(x10, last)
    x12 = fork(x7, x9, x11)
    x13 = lbind(sfilter, x1)
    x14 = chain(size, x13, x12)
    x15 = asindices(I)
    x16 = corners(x15)
    x17 = argmax(x16, x14)
    x18 = mostcolor(I)
    x19 = initset(x17)
    x20 = fill(x6, x18, x19)
    return x20