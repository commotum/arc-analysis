import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1f876c06(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = compose(last, first)
    x2 = power(last, TWO)
    x3 = fork(connect, x1, x2)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x0)
    x6 = paint(I, x5)
    return x6