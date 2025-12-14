import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6cdd2623(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = box(x0)
    x2 = rbind(difference, x1)
    x3 = chain(size, x2, toindices)
    x4 = matcher(x3, ZERO)
    x5 = partition(I)
    x6 = sfilter(x5, x4)
    x7 = argmax(x6, size)
    x8 = color(x7)
    x9 = toindices(x7)
    x10 = fork(either, hline, vline)
    x11 = prapply(connect, x9, x9)
    x12 = compose(flip, x4)
    x13 = fork(both, x12, x10)
    x14 = mfilter(x11, x13)
    x15 = mostcolor(I)
    x16 = shape(I)
    x17 = canvas(x15, x16)
    x18 = fill(x17, x8, x14)
    return x18