import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6e02f1e3(I: Grid) -> Grid:
    x0 = numcolors(I)
    x1 = equality(x0, THREE)
    x2 = height(I)
    x3 = decrement(x2)
    x4 = toivec(x3)
    x5 = branch(x1, x4, ORIGIN)
    x6 = equality(x0, TWO)
    x7 = shape(I)
    x8 = decrement(x7)
    x9 = width(I)
    x10 = decrement(x9)
    x11 = tojvec(x10)
    x12 = branch(x6, x8, x11)
    x13 = shape(I)
    x14 = canvas(ZERO, x13)
    x15 = connect(x5, x12)
    x16 = fill(x14, FIVE, x15)
    return x16