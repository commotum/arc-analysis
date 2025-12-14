import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f8ff0b80(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = order(x0, size)
    x2 = apply(color, x1)
    x3 = rbind(canvas, UNITY)
    x4 = apply(x3, x2)
    x5 = merge(x4)
    x6 = hmirror(x5)
    return x6