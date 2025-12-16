import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d037b0a7(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = rbind(shoot, DOWN)
    x3 = compose(x2, last)
    x4 = fork(recolor, first, x3)
    x5 = mapply(x4, x1)
    x6 = paint(I, x5)
    return x6