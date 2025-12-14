import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e76a88a6(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = argmax(x0, numcolors)
    x2 = normalize(x1)
    x3 = remove(x1, x0)
    x4 = apply(ulcorner, x3)
    x5 = lbind(shift, x2)
    x6 = mapply(x5, x4)
    x7 = paint(I, x6)
    return x7