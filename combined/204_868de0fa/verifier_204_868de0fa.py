import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_868de0fa(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = sfilter(x0, square)
    x2 = compose(even, height)
    x3 = sfilter(x1, x2)
    x4 = difference(x1, x3)
    x5 = merge(x3)
    x6 = merge(x4)
    x7 = fill(I, TWO, x5)
    x8 = fill(x7, SEVEN, x6)
    return x8