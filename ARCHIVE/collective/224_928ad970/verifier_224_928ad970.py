import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_928ad970(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = leastcolor(I)
    x3 = palette(I)
    x4 = remove(x2, x3)
    x5 = mostcolor(I)
    x6 = other(x4, x5)
    x7 = inbox(x1)
    x8 = fill(I, x6, x7)
    return x8