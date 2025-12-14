import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_794b24be(I: Grid) -> Grid:
    x0 = palette(I)
    x1 = remove(ONE, x0)
    x2 = lbind(colorcount, I)
    x3 = argmax(x1, x2)
    x4 = canvas(x3, THREE_BY_THREE)
    x5 = colorcount(I, ONE)
    x6 = decrement(x5)
    x7 = tojvec(x6)
    x8 = connect(ORIGIN, x7)
    x9 = fill(x4, TWO, x8)
    x10 = initset(UNITY)
    x11 = equality(x5, FOUR)
    x12 = branch(x11, x10, x8)
    x13 = fill(x9, TWO, x12)
    return x13