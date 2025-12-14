import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_2dee498d(I: Grid) -> Grid:
    x0 = hsplit(I, THREE)
    x1 = first(x0)
    return x1