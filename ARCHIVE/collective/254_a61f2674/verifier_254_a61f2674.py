import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a61f2674(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = argmax(x0, size)
    x2 = argmin(x0, size)
    x3 = merge(x0)
    x4 = cover(I, x3)
    x5 = fill(x4, ONE, x1)
    x6 = fill(x5, TWO, x2)
    return x6