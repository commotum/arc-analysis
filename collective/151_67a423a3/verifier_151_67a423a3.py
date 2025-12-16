import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_67a423a3(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = palette(I)
    x2 = remove(x0, x1)
    x3 = totuple(x2)
    x4 = first(x3)
    x5 = last(x3)
    x6 = ofcolor(I, x4)
    x7 = backdrop(x6)
    x8 = ofcolor(I, x5)
    x9 = backdrop(x8)
    x10 = intersection(x7, x9)
    x11 = outbox(x10)
    x12 = fill(I, FOUR, x11)
    return x12