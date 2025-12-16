import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_9565186b(I: Grid) -> Grid:
    x0 = shape(I)
    x1 = partition(I)
    x2 = argmax(x1, size)
    x3 = canvas(FIVE, x0)
    x4 = paint(x3, x2)
    return x4