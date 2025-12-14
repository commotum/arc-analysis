import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ded97339(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = lbind(recolor, NEG_ONE)
    x2 = rbind(ofcolor, x0)
    x3 = chain(x1, backdrop, x2)
    x4 = fork(paint, identity, x3)
    x5 = height(I)
    x6 = vsplit(I, x5)
    x7 = mapply(x4, x6)
    x8 = ofcolor(x7, NEG_ONE)
    x9 = dmirror(I)
    x10 = width(I)
    x11 = vsplit(x9, x10)
    x12 = mapply(x4, x11)
    x13 = dmirror(x12)
    x14 = ofcolor(x13, NEG_ONE)
    x15 = combine(x8, x14)
    x16 = fill(I, x0, x15)
    return x16