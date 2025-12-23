# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "50846271"
SERIAL = "118"
URL    = "https://arcprize.org/play?task=50846271"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_completion",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0, 0],
    [0, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5],
    [5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 5, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0],
    [5, 0, 0, 5, 5, 0, 0, 0, 5, 0, 0, 0, 2, 5, 5, 5, 0, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 2, 5, 5, 0, 0, 5, 0, 5, 5, 0],
    [0, 5, 0, 0, 5, 0, 0, 0, 5, 2, 5, 2, 5, 5, 5, 2, 5, 0, 5, 0, 0, 0],
    [0, 5, 5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 2, 5, 0, 0, 5, 0, 0, 5, 5, 5],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 5, 0, 2, 5, 0, 5, 5, 0, 5, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5, 0, 5],
    [5, 0, 0, 5, 0, 0, 0, 0, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 5, 0, 5, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 5, 0, 5, 5, 5, 0, 5, 5],
    [0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 5, 5],
    [0, 0, 5, 5, 0, 2, 5, 5, 5, 0, 0, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0],
    [5, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 0, 5],
    [0, 0, 2, 5, 5, 2, 2, 2, 2, 0, 0, 0, 5, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [0, 5, 5, 0, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 5, 2, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 5, 5, 0, 2, 5, 0, 0, 0, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 5],
    [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 0, 0, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 0, 0, 0, 5],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0, 0],
    [0, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5],
    [5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 5, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0],
    [5, 0, 0, 5, 5, 0, 0, 0, 5, 0, 0, 0, 2, 5, 5, 5, 0, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 2, 5, 5, 0, 0, 5, 0, 5, 5, 0],
    [0, 5, 0, 0, 5, 0, 0, 0, 5, 2, 8, 2, 8, 8, 8, 2, 5, 0, 5, 0, 0, 0],
    [0, 5, 5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 2, 5, 0, 0, 5, 0, 0, 5, 5, 5],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 5, 0, 2, 5, 0, 5, 5, 0, 5, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5, 0, 5],
    [5, 0, 0, 5, 0, 0, 0, 0, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 5, 0, 5, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 5, 0, 5, 5, 5, 0, 5, 5],
    [0, 5, 5, 0, 0, 8, 0, 0, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 5, 5],
    [0, 0, 5, 5, 0, 2, 5, 5, 5, 0, 0, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0],
    [5, 0, 5, 0, 0, 8, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 0, 5],
    [0, 0, 2, 8, 8, 2, 2, 2, 2, 0, 0, 0, 5, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [0, 5, 5, 0, 0, 8, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 5, 2, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 5, 5, 0, 2, 5, 0, 0, 0, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 5],
    [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 0, 0, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 0, 0, 0, 5],
], dtype=int)

E2_IN = np.array([
    [0, 5, 0, 5, 0, 0, 0, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5],
    [5, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 0, 5, 5, 0],
    [0, 5, 0, 5, 5, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5, 5],
    [5, 0, 0, 5, 5, 0, 2, 5, 0, 5, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 2, 5, 2, 2, 2, 0, 5, 5, 0, 5, 0, 5, 5, 0, 0, 0],
    [5, 5, 0, 0, 5, 5, 2, 5, 5, 5, 0, 5, 0, 0, 5, 5, 0, 0, 0, 0],
    [0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5],
    [0, 0, 0, 5, 0, 5, 0, 5, 5, 5, 0, 5, 5, 5, 0, 0, 5, 5, 0, 5],
    [5, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5],
    [5, 0, 0, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5, 0, 5, 5, 5, 5, 5, 5],
    [5, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 2, 5, 2, 2, 2, 0, 0, 5],
    [0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5],
    [0, 0, 5, 0, 0, 5, 0, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
    [5, 5, 0, 0, 5, 5, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5, 5, 0, 0, 5, 5],
    [0, 0, 5, 0, 5, 5, 5, 2, 2, 5, 5, 0, 0, 5, 0, 0, 5, 5, 0, 0],
    [0, 5, 5, 0, 0, 5, 5, 2, 5, 0, 5, 5, 0, 0, 5, 0, 5, 5, 0, 0],
    [0, 0, 5, 0, 5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 0, 5, 0, 5, 5, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 0, 5, 5],
    [5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5],
], dtype=int)

E2_OUT = np.array([
    [0, 5, 0, 5, 0, 0, 0, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5],
    [5, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 0, 5, 5, 0],
    [0, 5, 0, 5, 5, 0, 8, 5, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5, 5],
    [5, 0, 0, 5, 5, 0, 2, 5, 0, 5, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 2, 8, 2, 2, 2, 0, 5, 5, 0, 5, 0, 5, 5, 0, 0, 0],
    [5, 5, 0, 0, 5, 5, 2, 5, 5, 5, 0, 5, 0, 0, 5, 5, 0, 0, 0, 0],
    [0, 0, 5, 5, 0, 0, 8, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5],
    [0, 0, 0, 5, 0, 5, 0, 5, 5, 5, 0, 5, 5, 5, 0, 0, 5, 5, 0, 5],
    [5, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 8, 0, 0, 5, 0, 5],
    [5, 0, 0, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5, 0, 8, 5, 5, 5, 5, 5],
    [5, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 2, 8, 2, 2, 2, 0, 0, 5],
    [0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 8, 0, 0, 5, 0, 5],
    [0, 0, 5, 0, 0, 5, 0, 5, 5, 0, 5, 5, 5, 5, 8, 5, 5, 5, 5, 0],
    [5, 5, 0, 0, 5, 5, 0, 8, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 0, 0, 0, 5, 5, 8, 0, 5, 5, 0, 5, 5, 5, 5, 0, 0, 5, 5],
    [0, 0, 5, 0, 5, 8, 8, 2, 2, 8, 5, 0, 0, 5, 0, 0, 5, 5, 0, 0],
    [0, 5, 5, 0, 0, 5, 5, 2, 5, 0, 5, 5, 0, 0, 5, 0, 5, 5, 0, 0],
    [0, 0, 5, 0, 5, 0, 5, 8, 0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 0, 5, 0, 5, 5, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 0, 5, 5],
    [5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5],
], dtype=int)

E3_IN = np.array([
    [0, 0, 5, 0, 5, 0, 5, 5, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0],
    [0, 0, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 5],
    [0, 5, 5, 5, 0, 5, 0, 5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 0, 0],
    [5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0],
    [5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 0, 5, 0, 0],
    [5, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 2, 5, 0, 0, 5, 0, 5, 5],
    [5, 0, 5, 0, 0, 5, 5, 0, 5, 2, 2, 5, 2, 2, 5, 5, 0, 5, 0],
    [0, 5, 0, 5, 5, 5, 5, 5, 0, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5],
    [5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 5, 2, 5, 5, 5, 0, 0, 0, 0],
    [5, 2, 2, 5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0],
    [5, 2, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5],
    [0, 2, 5, 0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0],
    [0, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 0, 0],
    [5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5],
    [0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 0, 5, 5, 5, 5, 0, 0, 0],
    [5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 0, 5, 0, 0, 0, 0],
    [5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0, 5],
    [0, 5, 5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 5, 5, 5],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 5, 0, 5, 0, 5, 5, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0],
    [0, 0, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 5],
    [0, 5, 5, 5, 0, 5, 0, 5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 0, 0],
    [5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0],
    [5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 5, 8, 5, 0, 0, 0, 5, 0, 0],
    [5, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 2, 5, 0, 0, 5, 0, 5, 5],
    [5, 0, 5, 0, 0, 5, 5, 0, 5, 2, 2, 8, 2, 2, 5, 5, 0, 5, 0],
    [0, 8, 0, 5, 5, 5, 5, 5, 0, 5, 0, 8, 5, 5, 5, 0, 5, 5, 5],
    [5, 8, 5, 0, 5, 5, 5, 5, 0, 0, 5, 2, 5, 5, 5, 0, 0, 0, 0],
    [8, 2, 2, 8, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0],
    [5, 2, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5],
    [0, 2, 5, 0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0],
    [0, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 0, 0],
    [5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5],
    [0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 0, 5, 5, 5, 5, 0, 0, 0],
    [5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 0, 5, 0, 0, 0, 0],
    [5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0, 5],
    [0, 5, 5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 5, 5, 5],
], dtype=int)

E4_IN = np.array([
    [0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5],
    [5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5],
    [5, 0, 5, 0, 0, 5, 5, 0, 2, 0, 5, 0],
    [5, 5, 0, 0, 5, 0, 5, 0, 2, 5, 0, 5],
    [5, 0, 0, 5, 5, 5, 2, 5, 2, 2, 2, 0],
    [5, 5, 5, 0, 5, 5, 0, 5, 2, 0, 0, 5],
    [5, 5, 5, 0, 5, 0, 0, 5, 5, 0, 0, 0],
    [5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [5, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0],
    [5, 0, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5],
], dtype=int)

E4_OUT = np.array([
    [0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5],
    [5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5],
    [5, 0, 5, 0, 0, 5, 5, 0, 2, 0, 5, 0],
    [5, 5, 0, 0, 5, 0, 5, 0, 2, 5, 0, 5],
    [5, 0, 0, 5, 5, 5, 2, 8, 2, 2, 2, 0],
    [5, 5, 5, 0, 5, 5, 0, 5, 2, 0, 0, 5],
    [5, 5, 5, 0, 5, 0, 0, 5, 8, 0, 0, 0],
    [5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [5, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0],
    [5, 0, 0, 0, 0, 5, 0, 0, 5, 5, 5, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 5, 0, 5, 0, 0, 5, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 0, 5, 5, 0],
    [0, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 5],
    [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 5],
    [0, 0, 5, 5, 0, 5, 5, 5, 0, 5, 0, 5, 0, 5, 0, 5, 5, 0, 5, 5, 5, 0],
    [0, 5, 0, 5, 2, 2, 5, 2, 2, 5, 0, 0, 5, 0, 5, 5, 5, 0, 0, 5, 5, 0],
    [0, 0, 0, 5, 0, 5, 2, 5, 5, 5, 0, 5, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [5, 5, 0, 0, 5, 5, 2, 0, 5, 5, 0, 0, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 2, 5, 2, 5, 5, 0, 0, 5, 5, 5, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 5, 0, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 2, 5, 0, 5, 0, 0, 5, 5, 0],
    [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 5, 0, 0, 0, 5, 5],
    [5, 5, 0, 0, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 5, 5, 0, 0, 5, 5, 0, 5],
    [0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 0, 5],
    [5, 5, 0, 5, 5, 2, 2, 2, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 0, 5, 5, 0],
    [5, 0, 0, 0, 5, 2, 5, 0, 5, 0, 5, 0, 5, 5, 5, 5, 0, 0, 0, 0, 5, 5],
    [5, 5, 5, 0, 0, 2, 0, 5, 5, 0, 0, 2, 2, 2, 2, 2, 5, 0, 5, 0, 5, 5],
    [5, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 5, 0, 5, 5, 5, 0, 5, 5, 0],
    [5, 5, 5, 5, 5, 0, 5, 0, 5, 5, 5, 5, 0, 5, 0, 0, 5, 5, 0, 5, 0, 5],
], dtype=int)

T_OUT = np.array([
    [0, 5, 0, 5, 0, 0, 5, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 0, 5, 5, 0],
    [0, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 5],
    [0, 0, 0, 0, 5, 5, 8, 0, 0, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 5],
    [0, 0, 5, 5, 0, 5, 8, 5, 0, 5, 0, 5, 0, 5, 0, 5, 5, 0, 5, 5, 5, 0],
    [0, 5, 0, 5, 2, 2, 8, 2, 2, 5, 0, 0, 5, 0, 5, 5, 5, 0, 0, 5, 5, 0],
    [0, 0, 0, 5, 0, 5, 2, 5, 5, 5, 0, 5, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [5, 5, 0, 0, 5, 5, 2, 0, 5, 5, 0, 0, 0, 8, 0, 0, 0, 5, 5, 5, 5, 5],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 8, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 2, 8, 2, 8, 8, 0, 0, 5, 5, 5, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 8, 0, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 2, 5, 0, 5, 0, 0, 5, 5, 0],
    [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 5, 0, 0, 0, 5, 5],
    [5, 5, 0, 0, 5, 8, 5, 0, 0, 5, 5, 0, 5, 0, 5, 5, 0, 0, 5, 5, 0, 5],
    [0, 0, 5, 5, 5, 8, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 0, 5],
    [5, 5, 0, 8, 8, 2, 2, 2, 5, 5, 5, 0, 5, 8, 5, 0, 5, 0, 0, 5, 5, 0],
    [5, 0, 0, 0, 5, 2, 5, 0, 5, 0, 5, 0, 5, 8, 5, 5, 0, 0, 0, 0, 5, 5],
    [5, 5, 5, 0, 0, 2, 0, 5, 5, 0, 0, 2, 2, 2, 2, 2, 5, 0, 5, 0, 5, 5],
    [5, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 8, 0, 5, 5, 5, 0, 5, 5, 0],
    [5, 5, 5, 5, 5, 0, 5, 0, 5, 5, 5, 5, 0, 8, 0, 0, 5, 5, 0, 5, 0, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 118")


# --- Code Golf Solution (Compressed) ---
def q(u):
    for f in (2, 3):
        a, e, n, *r = [{s - j * 1j for j, u in enumerate(u) for s, u in enumerate(u) if u >= f} for f in (2, 0, 5, 6)]
        for j in e:
            i = {s for s in e if abs(s - j) in (2, 0, 1, f)}
            r += [f | i for f in r if a - f > i]
        for f in r:
            if a - n < f:
                return [[u + 3 * (s - j * 1j in f & n) for s, u in enumerate(u)] for j, u in enumerate(u)]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

ContainerContainer = Container[Container]

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

def shift(
    patch: Patch,
    directions: IntegerTuple
) -> Patch:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)

def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

def neighbors(
    loc: IntegerTuple
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)

def fill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices """
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def paint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid """
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def connect(
    a: IntegerTuple,
    b: IntegerTuple
) -> Indices:
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()

def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

rng = []

def unifint(
    diff_lb: float,
    diff_ub: float,
    bounds: Tuple[int, int]
) -> int:
    """
    diff_lb: lower bound for difficulty, must be in range [0, diff_ub]
    diff_ub: upper bound for difficulty, must be in range [diff_lb, 1]
    bounds: interval [a, b] determining the integer values that can be sampled
    """
    a, b = bounds
    d = uniform(diff_lb, diff_ub)
    global rng
    rng.append(d)
    return min(max(a, round(a + (b - a) * d)), b)

def generate_50846271(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    cf1 = lambda d: {(d//2, 0), (d//2, d-1)} | set(sample(totuple(connect((d//2, 0), (d//2, d-1))), randint(1, d)))
    cf2 = lambda d: {(0, d//2), (d - 1, d//2)} | set(sample(totuple(connect((0, d//2), (d-1, d//2))), randint(1, d)))
    cf3 = lambda d: set(sample(totuple(remove((d//2, d//2), connect((d//2, 0), (d//2, d-1)))), randint(1, d-1))) | set(sample(totuple(remove((d//2, d//2), connect((0, d//2), (d - 1, d//2)))), randint(1, d-1)))
    cf = lambda d: choice((cf1, cf2, cf3))(d)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    dim = unifint(diff_lb, diff_ub, (1, 3))
    dim = 2 * dim + 1
    cross = connect((dim//2, 0), (dim//2, dim - 1)) | connect((0, dim//2), (dim - 1, dim//2))
    bgc, crossc, noisec = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    namt = unifint(diff_lb, diff_ub, (int(0.35 * h * w), int(0.65 * h * w)))
    inds = asindices(gi)
    noise = sample(totuple(inds), namt)
    gi = fill(gi, noisec, noise)
    initcross = choice((cf1, cf2))(dim)
    loci = randint(0, h - dim)
    locj = randint(0, w - dim)
    delt = shift(cross - initcross, (loci, locj))
    gi = fill(gi, crossc, shift(initcross, (loci, locj)))
    gi = fill(gi, noisec, delt)
    go = fill(gi, 8, delt)
    plcd = shift(cross, (loci, locj))
    bd = backdrop(plcd)
    nbhs = mapply(neighbors, plcd)
    inds = (inds - plcd) - nbhs
    nbhs2 = mapply(neighbors, nbhs)
    inds = inds - nbhs2
    inds = inds - mapply(neighbors, nbhs2)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) / (10 * dim)))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - dim and ij[1] <= w - dim)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        marked = shift(cf(dim), loc)
        full = shift(cross, loc)
        unmarked = full - marked
        inobj = recolor(noisec, unmarked) | recolor(crossc, marked)
        outobj = recolor(8, unmarked) | recolor(crossc, marked)
        outobji = toindices(outobj)
        if outobji.issubset(inds):
            dnbhs = mapply(neighbors, outobji)
            dnbhs2 = mapply(neighbors, dnbhs)
            inds = (inds - outobji) - (dnbhs | dnbhs2 | mapply(neighbors, dnbhs2))
            succ += 1
            gi = paint(gi, inobj)
            go = paint(go, outobj)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ONE = 1

TWO = 2

FOUR = 4

FIVE = 5

EIGHT = 8

T = True

DOWN = (1, 0)

RIGHT = (0, 1)

UP = (-1, 0)

LEFT = (0, -1)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

def rbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)

def lbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the leftmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)

def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

def height(
    piece: Piece
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def objects(
    grid: Grid,
    univalued: Boolean,
    diagonal: Boolean,
    without_bg: Boolean
) -> Objects:
    """ objects occurring on the grid """
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)

def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))

def lowermost(
    patch: Patch
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

def rightmost(
    patch: Patch
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))

def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1

def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_50846271(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = interval(TWO, FIVE, ONE)
    x3 = rbind(shift, RIGHT)
    x4 = rbind(shift, LEFT)
    x5 = rbind(shift, UP)
    x6 = rbind(shift, DOWN)
    x7 = lbind(fork, intersection)
    x8 = lbind(x7, identity)
    x9 = lbind(rbind, shift)
    x10 = compose(x8, x9)
    x11 = compose(x10, tojvec)
    x12 = chain(x10, tojvec, invert)
    x13 = compose(x10, toivec)
    x14 = chain(x10, toivec, invert)
    x15 = lbind(compose, initset)
    x16 = lbind(rbind, rapply)
    x17 = lbind(chain, first)
    x18 = lbind(compose, x4)
    x19 = x15(x11)
    x20 = rbind(x17, x19)
    x21 = chain(x18, x20, x16)
    x22 = lbind(compose, x3)
    x23 = x15(x12)
    x24 = rbind(x17, x23)
    x25 = chain(x22, x24, x16)
    x26 = lbind(compose, x5)
    x27 = x15(x13)
    x28 = rbind(x17, x27)
    x29 = chain(x26, x28, x16)
    x30 = lbind(compose, x6)
    x31 = x15(x14)
    x32 = rbind(x17, x31)
    x33 = chain(x30, x32, x16)
    x34 = rbind(ofcolor, x0)
    x35 = compose(x21, x34)
    x36 = compose(x25, x34)
    x37 = compose(x29, x34)
    x38 = compose(x33, x34)
    x39 = lbind(fork, combine)
    x40 = fork(x39, x35, x36)
    x41 = fork(x39, x37, x38)
    x42 = fork(x39, x40, x41)
    x43 = lbind(recolor, x0)
    x44 = rbind(mapply, x2)
    x45 = chain(x43, x44, x42)
    x46 = fork(paint, identity, x45)
    x47 = power(x46, FOUR)
    x48 = x47(I)
    x49 = objects(x48, T, T, T)
    x50 = colorfilter(x49, x0)
    x51 = compose(maximum, shape)
    x52 = apply(x51, x50)
    x53 = maximum(x52)
    x54 = ofcolor(x48, x0)
    x55 = rbind(contained, x54)
    x56 = rbind(add, RIGHT)
    x57 = compose(x55, x56)
    x58 = rbind(add, LEFT)
    x59 = compose(x55, x58)
    x60 = fork(either, x57, x59)
    x61 = rbind(add, DOWN)
    x62 = compose(x55, x61)
    x63 = rbind(add, UP)
    x64 = compose(x55, x63)
    x65 = fork(either, x62, x64)
    x66 = fork(both, x60, x65)
    x67 = matcher(size, x53)
    x68 = fork(either, vline, hline)
    x69 = fork(both, x67, x68)
    x70 = sfilter(x50, x69)
    x71 = apply(center, x70)
    x72 = sfilter(x54, x66)
    x73 = combine(x72, x71)
    x74 = halve(x53)
    x75 = invert(x74)
    x76 = toivec(x75)
    x77 = rbind(add, x76)
    x78 = toivec(x74)
    x79 = rbind(add, x78)
    x80 = fork(connect, x77, x79)
    x81 = invert(x74)
    x82 = tojvec(x81)
    x83 = rbind(add, x82)
    x84 = tojvec(x74)
    x85 = rbind(add, x84)
    x86 = fork(connect, x83, x85)
    x87 = fork(combine, x80, x86)
    x88 = mapply(x87, x73)
    x89 = fill(x48, x0, x88)
    x90 = replace(x89, x0, EIGHT)
    x91 = fill(x90, x0, x1)
    return x91


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_50846271(inp)
        assert pred == _to_grid(expected), f"{name} failed"
