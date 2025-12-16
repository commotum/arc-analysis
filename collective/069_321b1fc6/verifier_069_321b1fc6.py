import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_321b1fc6(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = argmax(x0, numcolors)
    x2 = remove(x1, x0)
    x3 = normalize(x1)
    x4 = apply(ulcorner, x2)
    x5 = lbind(shift, x3)
    x6 = mapply(x5, x4)
    x7 = paint(I, x6)
    x8 = cover(x7, x1)
    return x8