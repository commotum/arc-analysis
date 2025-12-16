import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_39a8645d(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = totuple(x0)
    x2 = apply(normalize, x1)
    x3 = mostcommon(x2)
    x4 = mostcolor(I)
    x5 = shape(x3)
    x6 = canvas(x4, x5)
    x7 = paint(x6, x3)
    return x7