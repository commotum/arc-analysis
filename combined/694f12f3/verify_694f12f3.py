import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_694f12f3(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = fork(multiply, height, width)
    x2 = fork(equality, size, x1)
    x3 = sfilter(x0, x2)
    x4 = compose(backdrop, inbox)
    x5 = argmin(x3, size)
    x6 = argmax(x3, size)
    x7 = x4(x5)
    x8 = x4(x6)
    x9 = fill(I, ONE, x7)
    x10 = fill(x9, TWO, x8)
    return x10