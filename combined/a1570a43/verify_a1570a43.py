import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a1570a43(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = fork(equality, toindices, corners)
    x3 = fork(multiply, height, width)
    x4 = sfilter(x0, x2)
    x5 = argmax(x4, x3)
    x6 = difference(x1, x5)
    x7 = mostcolor(I)
    x8 = fill(I, x7, x6)
    x9 = normalize(x6)
    x10 = ulcorner(x5)
    x11 = increment(x10)
    x12 = shift(x9, x11)
    x13 = paint(x8, x12)
    return x13