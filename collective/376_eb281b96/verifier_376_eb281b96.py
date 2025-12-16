import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_eb281b96(I: Grid) -> Grid:
    x0 = height(I)
    x1 = width(I)
    x2 = decrement(x0)
    x3 = astuple(x2, x1)
    x4 = crop(I, ORIGIN, x3)
    x5 = hmirror(x4)
    x6 = vconcat(I, x5)
    x7 = double(x2)
    x8 = astuple(x7, x1)
    x9 = crop(x6, DOWN, x8)
    x10 = vconcat(x6, x9)
    return x10