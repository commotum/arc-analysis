import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b94a9452(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(multiply, height, width)
    x2 = argmax(x0, x1)
    x3 = remove(x2, x0)
    x4 = merge(x3)
    x5 = subgrid(x4, I)
    x6 = mostcolor(x5)
    x7 = leastcolor(x5)
    x8 = switch(x5, x6, x7)
    return x8