import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a416b8f3(I: Grid) -> Grid:
    x0 = hconcat(I, I)
    return x0