import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1fad071e(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = colorfilter(x0, ONE)
    x2 = sizefilter(x1, FOUR)
    x3 = fork(equality, height, width)
    x4 = sfilter(x2, x3)
    x5 = size(x4)
    x6 = subtract(FIVE, x5)
    x7 = astuple(ONE, x5)
    x8 = canvas(ONE, x7)
    x9 = astuple(ONE, x6)
    x10 = mostcolor(I)
    x11 = canvas(x10, x9)
    x12 = hconcat(x8, x11)
    return x12