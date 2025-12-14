import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_10fcaaa3(I: Grid) -> Grid:
    x0 = hconcat(I, I)
    x1 = vconcat(x0, x0)
    x2 = asindices(x1)
    x3 = mostcolor(I)
    x4 = ofcolor(x1, x3)
    x5 = difference(x2, x4)
    x6 = mapply(ineighbors, x5)
    x7 = underfill(x1, EIGHT, x6)
    return x7