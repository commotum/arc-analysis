import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a5f85a15(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = compose(increment, double)
    x3 = shoot(ORIGIN, UNITY)
    x4 = apply(x2, x3)
    x5 = order(x4, identity)
    x6 = lbind(contained, ZERO)
    x7 = sfilter(x1, x6)
    x8 = lbind(shift, x5)
    x9 = mapply(x8, x7)
    x10 = fill(I, FOUR, x9)
    return x10