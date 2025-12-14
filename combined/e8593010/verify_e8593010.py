import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e8593010(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = sizefilter(x0, ONE)
    x2 = sizefilter(x0, TWO)
    x3 = sizefilter(x0, THREE)
    x4 = merge(x1)
    x5 = fill(I, THREE, x4)
    x6 = merge(x2)
    x7 = fill(x5, TWO, x6)
    x8 = merge(x3)
    x9 = fill(x7, ONE, x8)
    return x9