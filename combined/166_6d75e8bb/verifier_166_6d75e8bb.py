import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6d75e8bb(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = argmin(x0, size)
    x2 = delta(x1)
    x3 = fill(I, TWO, x2)
    return x3