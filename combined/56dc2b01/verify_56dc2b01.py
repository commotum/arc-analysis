import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_56dc2b01(I: Grid) -> Grid:
    x0 = ofcolor(I, TWO)
    x1 = hline(x0)
    x2 = branch(x1, dmirror, identity)
    x3 = x2(I)
    x4 = fgpartition(x3)
    x5 = matcher(color, TWO)
    x6 = compose(flip, x5)
    x7 = extract(x4, x6)
    x8 = ofcolor(x3, TWO)
    x9 = leftmost(x8)
    x10 = leftmost(x7)
    x11 = greater(x9, x10)
    x12 = manhattan(x7, x8)
    x13 = decrement(x12)
    x14 = branch(x11, identity, invert)
    x15 = branch(x11, decrement, increment)
    x16 = branch(x11, leftmost, rightmost)
    x17 = x14(x13)
    x18 = tojvec(x17)
    x19 = shift(x7, x18)
    x20 = x16(x19)
    x21 = x15(x20)
    x22 = tojvec(x21)
    x23 = vfrontier(x22)
    x24 = cover(x3, x7)
    x25 = paint(x24, x19)
    x26 = fill(x25, EIGHT, x23)
    x27 = x2(x26)
    return x27