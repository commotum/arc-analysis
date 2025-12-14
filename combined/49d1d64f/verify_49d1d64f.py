import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_49d1d64f(I: Grid) -> Grid:
    x0 = shape(I)
    x1 = increment(x0)
    x2 = increment(x1)
    x3 = canvas(ZERO, x2)
    x4 = asobject(I)
    x5 = shift(x4, UNITY)
    x6 = shift(x5, LEFT)
    x7 = paint(x3, x6)
    x8 = shift(x5, RIGHT)
    x9 = paint(x7, x8)
    x10 = shift(x5, UP)
    x11 = paint(x9, x10)
    x12 = shift(x5, DOWN)
    x13 = paint(x11, x12)
    x14 = paint(x13, x5)
    return x14