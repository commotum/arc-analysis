import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_83302e8f(I: Grid) -> Grid:
    x0 = index(I, ORIGIN)
    x1 = objects(I, T, F, F)
    x2 = fork(multiply, height, width)
    x3 = fork(equality, size, x2)
    x4 = chain(positive, decrement, size)
    x5 = colorfilter(x1, x0)
    x6 = fork(both, x3, x4)
    x7 = sfilter(x5, x6)
    x8 = merge(x7)
    x9 = ofcolor(I, x0)
    x10 = fill(I, FOUR, x9)
    x11 = fill(x10, THREE, x8)
    return x11