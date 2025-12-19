import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b27ca6d3(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = sizefilter(x0, TWO)
    x2 = mapply(outbox, x1)
    x3 = fill(I, THREE, x2)
    return x3