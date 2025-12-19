import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_05f2a901(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = fork(multiply, height, width)
    x2 = fork(equality, size, x1)
    x3 = extract(x0, x2)
    x4 = other(x0, x3)
    x5 = gravitate(x4, x3)
    x6 = move(I, x4, x5)
    return x6