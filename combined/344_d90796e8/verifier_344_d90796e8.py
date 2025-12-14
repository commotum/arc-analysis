import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d90796e8(I: Grid) -> Grid:
    x0 = ofcolor(I, TWO)
    x1 = ofcolor(I, THREE)
    x2 = compose(positive, size)
    x3 = lbind(intersection, x1)
    x4 = chain(x2, x3, dneighbors)
    x5 = compose(positive, size)
    x6 = lbind(intersection, x0)
    x7 = chain(x5, x6, dneighbors)
    x8 = sfilter(x0, x4)
    x9 = sfilter(x1, x7)
    x10 = cover(I, x8)
    x11 = fill(x10, EIGHT, x9)
    return x11