import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5117e062(I: Grid) -> Grid:
    x0 = objects(I, F, T, T)
    x1 = argmax(x0, numcolors)
    x2 = mostcolor(x1)
    x3 = normalize(x1)
    x4 = mostcolor(I)
    x5 = shape(x1)
    x6 = canvas(x4, x5)
    x7 = fill(x6, x2, x3)
    return x7