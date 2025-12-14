import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1cf80156(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = palette(I)
    x4 = other(x3, x2)
    x5 = objects(I, T, T, F)
    x6 = matcher(color, x4)
    x7 = extract(x5, x6)
    x8 = subgrid(x7, I)
    return x8