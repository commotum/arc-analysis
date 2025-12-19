import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_60b61512(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = mapply(delta, x0)
    x2 = fill(I, SEVEN, x1)
    return x2