import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e9afcf9a(I: Grid) -> Grid:
    x0 = asobject(I)
    x1 = height(I)
    x2 = decrement(x1)
    x3 = lbind(subtract, x2)
    x4 = compose(double, halve)
    x5 = fork(equality, identity, x4)
    x6 = compose(last, last)
    x7 = chain(flip, x5, x6)
    x8 = sfilter(x0, x7)
    x9 = chain(x3, first, last)
    x10 = compose(last, last)
    x11 = fork(astuple, x9, x10)
    x12 = fork(astuple, first, x11)
    x13 = apply(x12, x8)
    x14 = paint(I, x13)
    return x14