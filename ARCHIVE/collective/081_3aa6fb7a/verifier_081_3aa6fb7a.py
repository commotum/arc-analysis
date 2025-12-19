import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3aa6fb7a(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = mapply(corners, x0)
    x2 = underfill(I, ONE, x1)
    return x2