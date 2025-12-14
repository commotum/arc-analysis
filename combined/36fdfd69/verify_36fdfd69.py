import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_36fdfd69(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = fork(subtract, first, last)
    x3 = fork(multiply, sign, identity)
    x4 = compose(x3, x2)
    x5 = lbind(greater, THREE)
    x6 = chain(x5, maximum, x4)
    x7 = lbind(lbind, astuple)
    x8 = rbind(chain, x7)
    x9 = lbind(compose, x6)
    x10 = rbind(x8, x9)
    x11 = lbind(lbind, sfilter)
    x12 = compose(x10, x11)
    x13 = lbind(mapply, backdrop)
    x14 = fork(apply, x12, identity)
    x15 = compose(x13, x14)
    x16 = power(x15, TWO)
    x17 = x16(x1)
    x18 = fill(I, FOUR, x17)
    x19 = fill(x18, x0, x1)
    return x19