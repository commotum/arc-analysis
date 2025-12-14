import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_be94b721(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = argmax(x0, size)
    x2 = color(x1)
    x3 = remove(x1, x0)
    x4 = argmax(x3, size)
    x5 = shape(x4)
    x6 = canvas(x2, x5)
    x7 = normalize(x4)
    x8 = paint(x6, x7)
    return x8