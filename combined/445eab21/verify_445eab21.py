import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_445eab21(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = fork(multiply, height, width)
    x2 = argmax(x0, x1)
    x3 = color(x2)
    x4 = canvas(x3, TWO_BY_TWO)
    return x4