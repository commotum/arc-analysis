import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b1948b0a(I: Grid) -> Grid:
    x0 = replace(I, SIX, TWO)
    return x0