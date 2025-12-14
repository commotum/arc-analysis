import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ea32f347(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = merge(x0)
    x2 = fill(I, FOUR, x1)
    x3 = argmin(x0, size)
    x4 = argmax(x0, size)
    x5 = fill(x2, ONE, x4)
    x6 = fill(x5, TWO, x3)
    return x6