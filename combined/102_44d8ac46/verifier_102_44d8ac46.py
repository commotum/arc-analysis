import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_44d8ac46(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = apply(delta, x0)
    x2 = mfilter(x1, square)
    x3 = fill(I, TWO, x2)
    return x3