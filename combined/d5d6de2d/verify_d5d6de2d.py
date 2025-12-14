import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d5d6de2d(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = merge(x0)
    x2 = cover(I, x1)
    x3 = mapply(delta, x0)
    x4 = fill(x2, THREE, x3)
    return x4