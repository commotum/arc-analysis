import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_31aa019c(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = first(x1)
    x3 = neighbors(x2)
    x4 = mostcolor(I)
    x5 = shape(I)
    x6 = canvas(x4, x5)
    x7 = initset(x2)
    x8 = fill(x6, x0, x7)
    x9 = fill(x8, TWO, x3)
    return x9