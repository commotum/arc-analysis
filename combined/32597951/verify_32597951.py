import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_32597951(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(multiply, height, width)
    x2 = argmin(x0, x1)
    x3 = delta(x2)
    x4 = fill(I, THREE, x3)
    return x4