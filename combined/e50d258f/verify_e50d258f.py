import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e50d258f(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = box(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = shape(I)
    x5 = canvas(x3, x4)
    x6 = hconcat(I, x5)
    x7 = objects(x6, F, F, T)
    x8 = rbind(colorcount, TWO)
    x9 = argmax(x7, x8)
    x10 = subgrid(x9, I)
    return x10