import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3bdb4ada(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(multiply, height, width)
    x2 = fork(equality, size, x1)
    x3 = compose(flip, x2)
    x4 = extract(x0, x3)
    x5 = remove(x4, x0)
    x6 = compose(flip, even)
    x7 = rbind(chain, first)
    x8 = rbind(chain, last)
    x9 = lbind(rbind, subtract)
    x10 = lbind(x7, x6)
    x11 = lbind(x8, x6)
    x12 = chain(x10, x9, uppermost)
    x13 = chain(x11, x9, leftmost)
    x14 = lbind(fork, both)
    x15 = fork(x14, x12, x13)
    x16 = fork(sfilter, toindices, x15)
    x17 = mapply(x16, x5)
    x18 = color(x4)
    x19 = fill(I, x18, x17)
    return x19