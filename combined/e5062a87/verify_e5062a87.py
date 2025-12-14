import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e5062a87(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = recolor(ZERO, x1)
    x3 = normalize(x2)
    x4 = occurrences(I, x3)
    x5 = toindices(x3)
    x6 = lbind(shift, x5)
    x7 = mapply(x6, x4)
    x8 = fill(I, x0, x7)
    return x8