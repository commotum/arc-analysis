import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_8e1813be(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = matcher(height, ONE)
    x2 = matcher(width, ONE)
    x3 = fork(either, x1, x2)
    x4 = sfilter(x0, x3)
    x5 = matcher(height, ONE)
    x6 = sfilter(x4, x5)
    x7 = size(x6)
    x8 = matcher(width, ONE)
    x9 = sfilter(x4, x8)
    x10 = size(x9)
    x11 = greater(x7, x10)
    x12 = branch(x11, dmirror, identity)
    x13 = branch(x11, uppermost, leftmost)
    x14 = order(x4, x13)
    x15 = apply(color, x14)
    x16 = size(x4)
    x17 = repeat(x15, x16)
    x18 = x12(x17)
    return x18