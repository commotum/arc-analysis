import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a9f96cdd(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = shift(x1, NEG_UNITY)
    x3 = recolor(THREE, x2)
    x4 = shift(x1, UNITY)
    x5 = recolor(SEVEN, x4)
    x6 = shift(x1, DOWN_LEFT)
    x7 = recolor(EIGHT, x6)
    x8 = shift(x1, UP_RIGHT)
    x9 = recolor(SIX, x8)
    x10 = mostcolor(I)
    x11 = fill(I, x10, x1)
    x12 = combine(x3, x5)
    x13 = combine(x7, x9)
    x14 = combine(x12, x13)
    x15 = paint(x11, x14)
    return x15