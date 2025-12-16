import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_178fcbfb(I: Grid) -> Grid:
    x0 = ofcolor(I, TWO)
    x1 = ofcolor(I, THREE)
    x2 = ofcolor(I, ONE)
    x3 = mapply(vfrontier, x0)
    x4 = mapply(hfrontier, x1)
    x5 = mapply(hfrontier, x2)
    x6 = fill(I, TWO, x3)
    x7 = fill(x6, THREE, x4)
    x8 = fill(x7, ONE, x5)
    return x8