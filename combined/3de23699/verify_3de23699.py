import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3de23699(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = corners(x0)
    x2 = mapply(dneighbors, x1)
    x3 = toobject(x2, I)
    x4 = mostcolor(x3)
    x5 = palette(I)
    x6 = remove(x4, x5)
    x7 = order(x6, identity)
    x8 = first(x7)
    x9 = last(x7)
    x10 = ofcolor(I, x8)
    x11 = ofcolor(I, x9)
    x12 = switch(I, x9, x8)
    x13 = combine(x10, x11)
    x14 = subgrid(x13, x12)
    x15 = trim(x14)
    return x15