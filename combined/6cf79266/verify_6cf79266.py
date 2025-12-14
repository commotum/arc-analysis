import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6cf79266(I: Grid) -> Grid:
    x0 = canvas(ZERO, THREE_BY_THREE)
    x1 = asobject(x0)
    x2 = occurrences(I, x1)
    x3 = lbind(shift, x1)
    x4 = mapply(x3, x2)
    x5 = fill(I, ONE, x4)
    return x5