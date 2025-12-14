import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5c0a986e(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = colorfilter(x0, TWO)
    x2 = colorfilter(x0, ONE)
    x3 = lbind(recolor, TWO)
    x4 = rbind(shoot, UNITY)
    x5 = chain(x3, x4, lrcorner)
    x6 = lbind(recolor, ONE)
    x7 = rbind(shoot, NEG_UNITY)
    x8 = chain(x6, x7, ulcorner)
    x9 = mapply(x5, x1)
    x10 = mapply(x8, x2)
    x11 = combine(x9, x10)
    x12 = paint(I, x11)
    return x12