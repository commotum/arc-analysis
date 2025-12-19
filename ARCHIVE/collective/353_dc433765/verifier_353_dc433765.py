import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_dc433765(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = palette(I)
    x2 = remove(x0, x1)
    x3 = other(x2, FOUR)
    x4 = ofcolor(I, x3)
    x5 = ofcolor(I, FOUR)
    x6 = center(x4)
    x7 = center(x5)
    x8 = subtract(x7, x6)
    x9 = sign(x8)
    x10 = recolor(x3, x4)
    x11 = move(I, x10, x9)
    return x11