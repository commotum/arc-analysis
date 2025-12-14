import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ce22a75a(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = apply(initset, x1)
    x3 = apply(outbox, x2)
    x4 = mapply(backdrop, x3)
    x5 = fill(I, ONE, x4)
    return x5