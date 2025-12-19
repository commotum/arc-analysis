import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_2bee17df(I: Grid) -> Grid:
    x0 = trim(I)
    x1 = mostcolor(x0)
    x2 = repeat(x1, ONE)
    x3 = lbind(repeat, THREE)
    x4 = compose(x3, size)
    x5 = matcher(dedupe, x2)
    x6 = rbind(branch, identity)
    x7 = rbind(x6, x4)
    x8 = compose(x7, x5)
    x9 = compose(initset, x8)
    x10 = fork(rapply, x9, identity)
    x11 = compose(first, x10)
    x12 = apply(x11, x0)
    x13 = dmirror(x0)
    x14 = apply(x11, x13)
    x15 = dmirror(x14)
    x16 = ofcolor(x12, THREE)
    x17 = ofcolor(x15, THREE)
    x18 = combine(x16, x17)
    x19 = shift(x18, UNITY)
    x20 = fill(I, THREE, x19)
    return x20