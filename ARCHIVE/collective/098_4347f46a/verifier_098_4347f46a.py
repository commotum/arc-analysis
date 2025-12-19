import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_4347f46a(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = fork(difference, toindices, box)
    x2 = mapply(x1, x0)
    x3 = mostcolor(I)
    x4 = fill(I, x3, x2)
    return x4