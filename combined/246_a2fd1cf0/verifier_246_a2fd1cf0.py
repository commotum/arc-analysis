import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a2fd1cf0(I: Grid) -> Grid:
    x0 = ofcolor(I, TWO)
    x1 = ofcolor(I, THREE)
    x2 = uppermost(x0)
    x3 = leftmost(x0)
    x4 = uppermost(x1)
    x5 = leftmost(x1)
    x6 = astuple(x2, x4)
    x7 = minimum(x6)
    x8 = maximum(x6)
    x9 = astuple(x7, x5)
    x10 = astuple(x8, x5)
    x11 = connect(x9, x10)
    x12 = astuple(x3, x5)
    x13 = minimum(x12)
    x14 = maximum(x12)
    x15 = astuple(x2, x13)
    x16 = astuple(x2, x14)
    x17 = connect(x15, x16)
    x18 = combine(x11, x17)
    x19 = underfill(I, EIGHT, x18)
    return x19