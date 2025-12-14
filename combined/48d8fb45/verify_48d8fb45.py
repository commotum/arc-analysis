import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_48d8fb45(I: Grid) -> Grid:
    x0 = objects(I, F, T, T)
    x1 = argmax(x0, numcolors)
    x2 = mostcolor(x1)
    x3 = matcher(first, x2)
    x4 = sfilter(x1, x3)
    x5 = shape(x4)
    x6 = normalize(x4)
    x7 = mostcolor(I)
    x8 = canvas(x7, x5)
    x9 = paint(x8, x6)
    return x9