import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d0f5fe59(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = size(x0)
    x2 = astuple(x1, x1)
    x3 = mostcolor(I)
    x4 = canvas(x3, x2)
    x5 = shoot(ORIGIN, UNITY)
    x6 = leastcolor(I)
    x7 = fill(x4, x6, x5)
    return x7