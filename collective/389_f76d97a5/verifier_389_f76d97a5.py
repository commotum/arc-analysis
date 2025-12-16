import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f76d97a5(I: Grid) -> Grid:
    x0 = palette(I)
    x1 = remove(FIVE, x0)
    x2 = first(x1)
    x3 = ofcolor(I, x2)
    x4 = fill(I, ZERO, x3)
    x5 = ofcolor(I, FIVE)
    x6 = fill(x4, x2, x5)
    return x6