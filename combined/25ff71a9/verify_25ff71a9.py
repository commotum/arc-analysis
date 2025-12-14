import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_25ff71a9(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = first(x0)
    x2 = move(I, x1, DOWN)
    return x2