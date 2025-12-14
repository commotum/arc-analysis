import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_28bf18c6(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = first(x0)
    x2 = subgrid(x1, I)
    x3 = hconcat(x2, x2)
    return x3