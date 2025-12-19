import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_963e52fc(I: Grid) -> Grid:
    x0 = width(I)
    x1 = asobject(I)
    x2 = hperiod(x1)
    x3 = height(x1)
    x4 = astuple(x3, x2)
    x5 = ulcorner(x1)
    x6 = crop(I, x5, x4)
    x7 = rot90(x6)
    x8 = double(x0)
    x9 = divide(x8, x2)
    x10 = increment(x9)
    x11 = repeat(x7, x10)
    x12 = merge(x11)
    x13 = rot270(x12)
    x14 = astuple(x3, x8)
    x15 = crop(x13, ORIGIN, x14)
    return x15