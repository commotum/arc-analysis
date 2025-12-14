import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_40853293(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(recolor, color, backdrop)
    x2 = apply(x1, x0)
    x3 = mfilter(x2, hline)
    x4 = mfilter(x2, vline)
    x5 = paint(I, x3)
    x6 = paint(x5, x4)
    return x6