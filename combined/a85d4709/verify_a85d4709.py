import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a85d4709(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = height(I)
    x2 = vsplit(I, x1)
    x3 = rbind(ofcolor, x0)
    x4 = compose(leftmost, x3)
    x5 = width(I)
    x6 = divide(x5, THREE)
    x7 = multiply(x6, TWO)
    x8 = lbind(greater, x6)
    x9 = compose(x8, x4)
    x10 = lbind(greater, x7)
    x11 = compose(x10, x4)
    x12 = compose(flip, x9)
    x13 = fork(both, x11, x12)
    x14 = fork(either, x9, x13)
    x15 = compose(flip, x14)
    x16 = rbind(multiply, TWO)
    x17 = compose(x16, x9)
    x18 = rbind(multiply, FOUR)
    x19 = compose(x18, x13)
    x20 = rbind(multiply, THREE)
    x21 = compose(x20, x15)
    x22 = fork(add, x17, x19)
    x23 = fork(add, x22, x21)
    x24 = width(I)
    x25 = rbind(repeat, x24)
    x26 = compose(x25, x23)
    x27 = apply(x26, x2)
    return x27