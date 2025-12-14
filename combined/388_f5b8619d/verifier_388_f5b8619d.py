import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f5b8619d(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = mapply(toindices, x0)
    x2 = mapply(vfrontier, x1)
    x3 = underfill(I, EIGHT, x2)
    x4 = hconcat(x3, x3)
    x5 = vconcat(x4, x4)
    return x5