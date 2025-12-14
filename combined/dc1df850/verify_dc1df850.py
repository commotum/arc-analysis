import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_dc1df850(I: Grid) -> Grid:
    x0 = ofcolor(I, TWO)
    x1 = mapply(neighbors, x0)
    x2 = underfill(I, ONE, x1)
    return x2