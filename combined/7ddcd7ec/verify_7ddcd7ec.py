import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_7ddcd7ec(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = argmax(x0, size)
    x2 = remove(x1, x0)
    x3 = lbind(position, x1)
    x4 = fork(shoot, center, x3)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x2)
    x7 = paint(I, x6)
    return x7