import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_c0f76784(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = mostcolor(I)
    x2 = colorfilter(x0, x1)
    x3 = sizefilter(x2, ONE)
    x4 = merge(x3)
    x5 = sizefilter(x2, FOUR)
    x6 = merge(x5)
    x7 = sizefilter(x2, NINE)
    x8 = merge(x7)
    x9 = fill(I, SIX, x4)
    x10 = fill(x9, SEVEN, x6)
    x11 = fill(x10, EIGHT, x8)
    return x11