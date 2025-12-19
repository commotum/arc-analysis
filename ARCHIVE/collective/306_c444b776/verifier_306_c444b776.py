import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_c444b776(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = leastcolor(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = objects(x5, F, F, T)
    x7 = argmax(x6, numcolors)
    x8 = apply(ulcorner, x6)
    x9 = normalize(x7)
    x10 = lbind(shift, x9)
    x11 = mapply(x10, x8)
    x12 = paint(I, x11)
    return x12