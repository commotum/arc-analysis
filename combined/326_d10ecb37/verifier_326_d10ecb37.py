import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d10ecb37(I: Grid) -> Grid:
    x0 = crop(I, ORIGIN, TWO_BY_TWO)
    return x0