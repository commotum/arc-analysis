import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_46442a0e(I: Grid) -> Grid:
    x0 = rot90(I)
    x1 = rot180(I)
    x2 = rot270(I)
    x3 = hconcat(I, x0)
    x4 = hconcat(x2, x1)
    x5 = vconcat(x3, x4)
    return x5