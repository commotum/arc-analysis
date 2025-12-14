import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5582e5ca(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = shape(I)
    x2 = canvas(x0, x1)
    return x2