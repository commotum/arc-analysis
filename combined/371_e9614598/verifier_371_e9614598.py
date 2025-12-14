import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e9614598(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = center(x1)
    x3 = dneighbors(x2)
    x4 = insert(x2, x3)
    x5 = fill(I, THREE, x4)
    return x5