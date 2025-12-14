import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_dbc1a6ce(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = lbind(recolor, EIGHT)
    x3 = rbind(ofcolor, x0)
    x4 = chain(x2, backdrop, x3)
    x5 = fork(paint, identity, x4)
    x6 = height(I)
    x7 = vsplit(I, x6)
    x8 = mapply(x5, x7)
    x9 = ofcolor(x8, EIGHT)
    x10 = dmirror(I)
    x11 = width(I)
    x12 = vsplit(x10, x11)
    x13 = mapply(x5, x12)
    x14 = dmirror(x13)
    x15 = ofcolor(x14, EIGHT)
    x16 = combine(x9, x15)
    x17 = difference(x16, x1)
    x18 = fill(I, EIGHT, x17)
    return x18