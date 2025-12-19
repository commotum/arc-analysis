import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_0ca9ddb6(I: Grid) -> Grid:
    x0 = ofcolor(I, ONE)
    x1 = ofcolor(I, TWO)
    x2 = mapply(dneighbors, x0)
    x3 = mapply(ineighbors, x1)
    x4 = fill(I, SEVEN, x2)
    x5 = fill(x4, FOUR, x3)
    return x5