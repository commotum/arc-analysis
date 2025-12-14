import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_42a50994(I: Grid) -> Grid:
    x0 = objects(I, F, T, T)
    x1 = sizefilter(x0, ONE)
    x2 = merge(x1)
    x3 = cover(I, x2)
    return x3