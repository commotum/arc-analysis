import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ae4f1146(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = box(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = objects(I, F, F, T)
    x5 = rbind(colorcount, ONE)
    x6 = argmax(x4, x5)
    x7 = subgrid(x6, I)
    return x7