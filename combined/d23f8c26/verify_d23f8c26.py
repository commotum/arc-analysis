import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d23f8c26(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = matcher(first, x0)
    x2 = compose(flip, x1)
    x3 = width(I)
    x4 = halve(x3)
    x5 = compose(last, last)
    x6 = matcher(x5, x4)
    x7 = compose(flip, x6)
    x8 = asobject(I)
    x9 = fork(both, x2, x7)
    x10 = sfilter(x8, x9)
    x11 = fill(I, x0, x10)
    return x11