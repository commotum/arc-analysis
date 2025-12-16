import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_c1d99e64(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = fill(I, TWO, x1)
    return x2