import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b60334d2(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = mostcolor(I)
    x2 = ofcolor(I, x0)
    x3 = replace(I, x0, x1)
    x4 = mapply(dneighbors, x2)
    x5 = mapply(ineighbors, x2)
    x6 = fill(x3, ONE, x4)
    x7 = fill(x6, x0, x5)
    return x7