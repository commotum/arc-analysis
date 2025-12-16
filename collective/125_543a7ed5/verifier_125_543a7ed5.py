import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_543a7ed5(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = mapply(outbox, x0)
    x2 = fill(I, THREE, x1)
    x3 = mapply(delta, x0)
    x4 = fill(x2, FOUR, x3)
    return x4