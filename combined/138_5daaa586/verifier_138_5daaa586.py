import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5daaa586(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, x0)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    x6 = outbox(x5)
    x7 = subgrid(x6, I)
    x8 = trim(x7)
    x9 = palette(x8)
    x10 = matcher(identity, x0)
    x11 = argmin(x9, x10)
    x12 = trim(x7)
    x13 = ofcolor(x12, x11)
    x14 = shift(x13, UNITY)
    x15 = ofcolor(x7, x11)
    x16 = difference(x15, x14)
    x17 = compose(first, first)
    x18 = compose(first, last)
    x19 = fork(equality, x17, x18)
    x20 = compose(last, first)
    x21 = compose(last, last)
    x22 = fork(equality, x20, x21)
    x23 = fork(either, x19, x22)
    x24 = product(x14, x16)
    x25 = sfilter(x24, x23)
    x26 = fork(connect, first, last)
    x27 = mapply(x26, x25)
    x28 = fill(x7, x11, x27)
    return x28