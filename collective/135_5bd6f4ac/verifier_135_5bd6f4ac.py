import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5bd6f4ac(I: Grid) -> Grid:
    x0 = rot270(I)
    x1 = crop(x0, ORIGIN, THREE_BY_THREE)
    x2 = rot90(x1)
    return x2