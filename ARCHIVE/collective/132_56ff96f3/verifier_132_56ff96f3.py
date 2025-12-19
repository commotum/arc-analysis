import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_56ff96f3(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = fork(recolor, color, backdrop)
    x2 = mapply(x1, x0)
    x3 = paint(I, x2)
    return x3