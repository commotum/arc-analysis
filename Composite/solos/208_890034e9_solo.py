# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "890034e9"
SERIAL = "208"
URL    = "https://arcprize.org/play?task=890034e9"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_repetition",
    "rectangle_guessing",
    "contouring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 8, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 8, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 2, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 8, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 2, 0, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 8, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 1, 0, 1, 0, 0, 1, 1, 8, 0, 0, 8],
    [0, 1, 8, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 8, 1, 1, 0, 0],
    [1, 1, 1, 8, 8, 1, 1, 1, 0, 0, 8, 1, 1, 1, 1, 1, 8, 1, 0, 0, 1],
    [8, 1, 0, 1, 1, 1, 1, 0, 8, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    [8, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 8, 1, 1, 8, 1],
    [1, 1, 1, 8, 1, 0, 1, 1, 8, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 8, 1, 1, 8, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 8, 1, 1, 1],
    [1, 1, 8, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 8, 1, 0, 1, 0, 1, 1, 8],
    [1, 1, 1, 1, 1, 1, 0, 0, 8, 1, 0, 0, 1, 1, 8, 1, 1, 8, 1, 0, 1],
    [8, 8, 8, 1, 1, 1, 1, 8, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 8, 0, 0, 8, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [1, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 8, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 8, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 8, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 8, 1, 8, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 8, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 8, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 2, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 8, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 2, 0, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 8, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 1, 0, 1, 0, 0, 1, 1, 8, 0, 0, 8],
    [0, 1, 8, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 8, 1, 1, 0, 0],
    [1, 1, 1, 8, 8, 1, 1, 1, 0, 0, 8, 1, 1, 1, 1, 1, 8, 1, 0, 0, 1],
    [8, 1, 0, 1, 1, 1, 1, 0, 8, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    [8, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 8, 1, 1, 8, 1],
    [1, 1, 1, 8, 1, 0, 1, 1, 8, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 8, 1, 1, 8, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 8, 1, 1, 1],
    [1, 1, 8, 1, 1, 1, 0, 1, 0, 2, 2, 2, 2, 8, 1, 0, 1, 0, 1, 1, 8],
    [1, 1, 1, 1, 1, 1, 0, 0, 8, 2, 0, 0, 2, 1, 8, 1, 1, 8, 1, 0, 1],
    [8, 8, 8, 1, 1, 1, 1, 8, 1, 2, 0, 0, 2, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 8, 0, 0, 8, 1, 2, 0, 0, 2, 1, 1, 1, 0, 1, 0, 1, 0],
    [1, 8, 8, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 8, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 8, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 8, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 8, 1, 8, 0],
], dtype=int)

E2_IN = np.array([
    [3, 0, 3, 4, 3, 3, 3, 3, 0, 3, 3, 4, 0, 3, 0, 4, 3, 4, 4, 0, 0],
    [3, 3, 0, 0, 3, 3, 3, 4, 0, 0, 4, 4, 4, 3, 0, 0, 3, 3, 4, 0, 3],
    [4, 4, 4, 3, 4, 3, 0, 3, 0, 0, 4, 3, 0, 3, 3, 4, 3, 0, 0, 3, 0],
    [0, 4, 4, 4, 3, 0, 3, 3, 3, 0, 3, 0, 3, 0, 0, 0, 0, 3, 4, 3, 3],
    [3, 3, 0, 4, 3, 3, 0, 0, 0, 0, 3, 0, 4, 4, 4, 3, 0, 3, 0, 0, 0],
    [0, 3, 0, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 3, 3, 3, 3, 4, 3, 0, 3],
    [0, 3, 0, 0, 3, 4, 0, 3, 4, 0, 4, 4, 0, 0, 3, 4, 0, 0, 0, 3, 3],
    [0, 3, 3, 3, 0, 4, 4, 3, 4, 3, 0, 3, 3, 3, 4, 0, 3, 0, 3, 3, 3],
    [4, 0, 4, 3, 4, 3, 4, 4, 0, 0, 4, 0, 0, 0, 0, 3, 0, 3, 3, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 3, 4, 4, 3, 4, 0, 0, 0, 4, 0, 0, 4, 3, 3],
    [3, 0, 0, 8, 8, 8, 8, 8, 4, 3, 0, 3, 3, 0, 4, 4, 0, 4, 4, 4, 4],
    [3, 3, 0, 8, 0, 0, 0, 8, 3, 0, 0, 0, 0, 4, 0, 3, 3, 0, 4, 3, 3],
    [0, 0, 0, 8, 0, 0, 0, 8, 3, 3, 0, 3, 3, 4, 3, 0, 4, 0, 3, 0, 0],
    [3, 0, 4, 8, 8, 8, 8, 8, 0, 3, 0, 3, 0, 0, 3, 3, 3, 0, 4, 3, 0],
    [4, 0, 0, 0, 0, 3, 0, 4, 0, 0, 3, 0, 0, 3, 3, 3, 4, 0, 4, 0, 3],
    [0, 0, 4, 3, 0, 0, 0, 3, 0, 0, 3, 4, 0, 0, 4, 0, 0, 3, 4, 3, 4],
    [4, 4, 0, 0, 3, 0, 3, 4, 4, 3, 4, 3, 4, 0, 4, 4, 0, 3, 4, 3, 4],
    [3, 4, 3, 3, 0, 0, 0, 0, 3, 0, 3, 4, 0, 0, 0, 3, 3, 3, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 3, 4, 0, 3, 3, 3, 4, 0, 4, 0, 3, 4, 0],
    [3, 3, 3, 0, 4, 0, 4, 3, 0, 0, 0, 3, 0, 0, 3, 3, 0, 0, 4, 3, 0],
    [0, 4, 3, 3, 3, 0, 4, 4, 3, 4, 3, 4, 0, 4, 3, 4, 4, 0, 0, 4, 0],
], dtype=int)

E2_OUT = np.array([
    [3, 0, 3, 4, 3, 3, 3, 3, 0, 3, 3, 4, 0, 3, 0, 4, 3, 4, 4, 0, 0],
    [3, 3, 0, 0, 3, 3, 3, 4, 0, 0, 4, 4, 4, 3, 0, 0, 3, 3, 4, 0, 3],
    [4, 4, 4, 3, 4, 3, 0, 3, 0, 0, 4, 3, 0, 3, 3, 4, 3, 0, 0, 3, 0],
    [0, 4, 4, 4, 3, 0, 3, 3, 3, 0, 3, 0, 3, 0, 0, 0, 0, 3, 4, 3, 3],
    [3, 3, 0, 4, 3, 3, 0, 0, 0, 0, 3, 0, 4, 4, 4, 3, 0, 3, 0, 0, 0],
    [0, 3, 0, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 3, 3, 3, 3, 4, 3, 0, 3],
    [0, 3, 0, 0, 3, 4, 0, 3, 4, 0, 4, 4, 0, 0, 3, 4, 0, 0, 0, 3, 3],
    [0, 3, 3, 3, 0, 4, 4, 3, 4, 3, 0, 8, 8, 8, 8, 8, 3, 0, 3, 3, 3],
    [4, 0, 4, 3, 4, 3, 4, 4, 0, 0, 4, 8, 0, 0, 0, 8, 0, 3, 3, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 3, 4, 4, 3, 8, 0, 0, 0, 8, 0, 0, 4, 3, 3],
    [3, 0, 0, 8, 8, 8, 8, 8, 4, 3, 0, 8, 8, 8, 8, 8, 0, 4, 4, 4, 4],
    [3, 3, 0, 8, 0, 0, 0, 8, 3, 0, 0, 0, 0, 4, 0, 3, 3, 0, 4, 3, 3],
    [0, 0, 0, 8, 0, 0, 0, 8, 3, 3, 0, 3, 3, 4, 3, 0, 4, 0, 3, 0, 0],
    [3, 0, 4, 8, 8, 8, 8, 8, 0, 3, 0, 3, 0, 0, 3, 3, 3, 0, 4, 3, 0],
    [4, 0, 0, 0, 0, 3, 0, 4, 0, 0, 3, 0, 0, 3, 3, 3, 4, 0, 4, 0, 3],
    [0, 0, 4, 3, 0, 0, 0, 3, 0, 0, 3, 4, 0, 0, 4, 0, 0, 3, 4, 3, 4],
    [4, 4, 0, 0, 3, 0, 3, 4, 4, 3, 4, 3, 4, 0, 4, 4, 0, 3, 4, 3, 4],
    [3, 4, 3, 3, 0, 0, 0, 0, 3, 0, 3, 4, 0, 0, 0, 3, 3, 3, 3, 0, 3],
    [0, 0, 0, 0, 0, 3, 0, 3, 3, 4, 0, 3, 3, 3, 4, 0, 4, 0, 3, 4, 0],
    [3, 3, 3, 0, 4, 0, 4, 3, 0, 0, 0, 3, 0, 0, 3, 3, 0, 0, 4, 3, 0],
    [0, 4, 3, 3, 3, 0, 4, 4, 3, 4, 3, 4, 0, 4, 3, 4, 4, 0, 0, 4, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 3, 0, 3, 2, 0, 2, 0, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
    [3, 2, 2, 0, 3, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 0, 3, 2],
    [3, 3, 0, 3, 0, 0, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 0, 0, 3, 2],
    [2, 2, 3, 2, 4, 4, 4, 4, 4, 4, 3, 0, 3, 2, 0, 2, 2, 2, 0, 0, 3],
    [3, 3, 2, 0, 4, 0, 0, 0, 0, 4, 2, 0, 2, 2, 0, 2, 3, 0, 2, 2, 0],
    [3, 2, 2, 2, 4, 0, 0, 0, 0, 4, 0, 3, 2, 2, 3, 2, 2, 3, 3, 2, 0],
    [2, 0, 2, 0, 4, 0, 0, 0, 0, 4, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2],
    [0, 2, 0, 2, 4, 4, 4, 4, 4, 4, 2, 2, 0, 2, 0, 2, 0, 0, 2, 2, 2],
    [2, 0, 2, 2, 2, 0, 2, 0, 2, 0, 3, 2, 3, 3, 0, 2, 0, 0, 0, 2, 2],
    [0, 2, 3, 0, 3, 0, 2, 3, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 3, 2, 0],
    [3, 0, 2, 0, 2, 0, 0, 2, 2, 0, 3, 3, 2, 3, 0, 3, 3, 0, 0, 3, 0],
    [2, 3, 0, 3, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 0, 2],
    [3, 2, 2, 0, 2, 0, 2, 2, 0, 3, 2, 2, 2, 2, 3, 0, 2, 2, 2, 2, 2],
    [3, 3, 3, 2, 0, 2, 0, 2, 0, 3, 2, 2, 2, 0, 0, 3, 2, 2, 3, 2, 2],
    [0, 0, 2, 2, 2, 3, 2, 0, 0, 2, 3, 2, 0, 3, 0, 2, 2, 3, 2, 2, 0],
    [2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 2, 3, 0],
    [2, 2, 2, 2, 3, 0, 0, 3, 3, 2, 0, 0, 0, 0, 0, 0, 2, 2, 3, 2, 0],
    [2, 0, 3, 2, 2, 2, 3, 2, 3, 3, 3, 0, 0, 0, 0, 0, 2, 0, 0, 2, 3],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 3, 2, 3, 2, 2, 3, 0, 0, 2, 2, 0, 0],
    [0, 3, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 3, 0, 2, 0, 0, 0, 3, 2],
    [2, 3, 2, 2, 2, 0, 0, 3, 2, 0, 3, 2, 0, 2, 2, 2, 3, 0, 0, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 3, 0, 3, 2, 0, 2, 0, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
    [3, 2, 2, 0, 3, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 0, 3, 2],
    [3, 3, 0, 3, 0, 0, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 0, 0, 3, 2],
    [2, 2, 3, 2, 4, 4, 4, 4, 4, 4, 3, 0, 3, 2, 0, 2, 2, 2, 0, 0, 3],
    [3, 3, 2, 0, 4, 0, 0, 0, 0, 4, 2, 0, 2, 2, 0, 2, 3, 0, 2, 2, 0],
    [3, 2, 2, 2, 4, 0, 0, 0, 0, 4, 0, 3, 2, 2, 3, 2, 2, 3, 3, 2, 0],
    [2, 0, 2, 0, 4, 0, 0, 0, 0, 4, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2],
    [0, 2, 0, 2, 4, 4, 4, 4, 4, 4, 2, 2, 0, 2, 0, 2, 0, 0, 2, 2, 2],
    [2, 0, 2, 2, 2, 0, 2, 0, 2, 0, 3, 2, 3, 3, 0, 2, 0, 0, 0, 2, 2],
    [0, 2, 3, 0, 3, 0, 2, 3, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 3, 2, 0],
    [3, 0, 2, 0, 2, 0, 0, 2, 2, 0, 3, 3, 2, 3, 0, 3, 3, 0, 0, 3, 0],
    [2, 3, 0, 3, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 0, 2],
    [3, 2, 2, 0, 2, 0, 2, 2, 0, 3, 2, 2, 2, 2, 3, 0, 2, 2, 2, 2, 2],
    [3, 3, 3, 2, 0, 2, 0, 2, 0, 3, 2, 2, 2, 0, 0, 3, 2, 2, 3, 2, 2],
    [0, 0, 2, 2, 2, 3, 2, 0, 0, 2, 4, 4, 4, 4, 4, 4, 2, 3, 2, 2, 0],
    [2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 4, 0, 0, 0, 0, 4, 0, 0, 2, 3, 0],
    [2, 2, 2, 2, 3, 0, 0, 3, 3, 2, 4, 0, 0, 0, 0, 4, 2, 2, 3, 2, 0],
    [2, 0, 3, 2, 2, 2, 3, 2, 3, 3, 4, 0, 0, 0, 0, 4, 2, 0, 0, 2, 3],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 4, 4, 0, 2, 2, 0, 0],
    [0, 3, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 3, 0, 2, 0, 0, 0, 3, 2],
    [2, 3, 2, 2, 2, 0, 0, 3, 2, 0, 3, 2, 0, 2, 2, 2, 3, 0, 0, 2, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 2, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 1, 1, 1, 0, 1, 2],
    [1, 1, 1, 0, 2, 1, 2, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 2, 1, 1],
    [1, 1, 1, 0, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1],
    [0, 2, 1, 0, 1, 1, 2, 2, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 2],
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 2, 0],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 2, 2, 1, 0, 1, 2, 2, 1, 1, 2, 0, 0, 1, 0, 1, 1, 1, 2],
    [1, 0, 1, 0, 1, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1],
    [1, 3, 3, 3, 3, 1, 2, 0, 2, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    [2, 3, 0, 0, 3, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 3, 0, 0, 3, 1, 1, 2, 0, 1, 1, 1, 0, 2, 1, 1, 1, 0, 1, 1, 1],
    [1, 3, 0, 0, 3, 1, 2, 0, 0, 0, 1, 2, 1, 1, 1, 2, 1, 0, 1, 0, 1],
    [1, 3, 0, 0, 3, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 3, 0, 0, 3, 1, 0, 2, 0, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1],
    [1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 2, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
], dtype=int)

T_OUT = np.array([
    [0, 2, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 1, 1, 1, 0, 1, 2],
    [1, 1, 1, 0, 2, 1, 2, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 2, 1, 1],
    [1, 1, 1, 0, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 3, 3, 0, 1, 1, 1, 1],
    [0, 2, 1, 0, 1, 1, 2, 2, 1, 1, 0, 1, 3, 0, 0, 3, 0, 1, 1, 1, 2],
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 3, 0, 0, 3, 1, 0, 0, 2, 0],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 3, 0, 0, 3, 1, 0, 1, 1, 1],
    [1, 1, 1, 2, 2, 1, 0, 1, 2, 2, 1, 1, 3, 0, 0, 3, 0, 1, 1, 1, 2],
    [1, 0, 1, 0, 1, 0, 0, 2, 1, 1, 1, 0, 3, 0, 0, 3, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 1, 1, 0, 1],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1],
    [1, 3, 3, 3, 3, 1, 2, 0, 2, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    [2, 3, 0, 0, 3, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 3, 0, 0, 3, 1, 1, 2, 0, 1, 1, 1, 0, 2, 1, 1, 1, 0, 1, 1, 1],
    [1, 3, 0, 0, 3, 1, 2, 0, 0, 0, 1, 2, 1, 1, 1, 2, 1, 0, 1, 0, 1],
    [1, 3, 0, 0, 3, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 3, 0, 0, 3, 1, 0, 2, 0, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1],
    [1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 2, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z=[r[:] for r in g]
 for s in R(min([h,w]),1,-1):
  t=0
  for r in R(h):
   for c in R(w):
    X=g[r:r+s]
    X=[m[c:c+s][:] for m in X]
    if sum(X,[]).count(0)==s*s:
     t=1
     for i in R(r,r+s):
      for j in R(c,c+s):
       Z[i][j]=9
  if t:return Z
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return eval((i := min((k := (str(g) + '#[]' * X)), key=k.count)).join(split(sub(i, f')[^{i}](', sub('[^%s]+' % i * 18, lambda x: '.' * len(x[0]), k)).strip('.()'), k)))


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

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

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

def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def occurrences(
    grid: Grid,
    obj: Object
) -> Indices:
    """ locations of occurrences of object in grid """
    occurrences = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    for i in range(h):
        for j in range(w):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if 0 <= a < h and 0 <= b < w:
                    if grid[a][b] != v:
                        occurs = False
                        break
                else:
                    occurs = False
                    break
            if occurs:
                occurrences.add((i, j))
    return frozenset(occurrences)

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

def generate_890034e9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = randint(2, h//4)
    ow = randint(2, w//4)
    markercol = choice(cols)
    remcols = remove(markercol, cols)
    numbgc = unifint(diff_lb, diff_ub, (1, 8))
    bgcols = sample(remcols, numbgc)
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    obj = {(choice(bgcols), ij) for ij in inds}
    gi = paint(gi, obj)
    numbl = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    blacks = sample(totuple(inds), numbl)
    gi = fill(gi, 0, blacks)
    patt = asindices(canvas(-1, (oh, ow)))
    tocover = set()
    for occ in occurrences(gi, recolor(0, patt)):
        tocover.add(choice(totuple(shift(patt, occ))))
    tocover = {(choice(bgcols), ij) for ij in tocover}
    gi = paint(gi, tocover)
    noccs = unifint(diff_lb, diff_ub, (2, (h * w) // ((oh + 2) * (ow + 2))))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        bd = shift(patt, loc)
        plcd = outbox(bd)
        if plcd.issubset(inds):
            succ += 1
            inds = inds - plcd
            gi = fill(gi, 0, bd)
            go = fill(go, 0, bd)
            if succ == 1:
                gi = fill(gi, markercol, plcd)
            go = fill(go, markercol, plcd)
            loci, locj = loc
            ln1 = connect((loci-1, locj), (loci-1, locj+ow-1))
            ln2 = connect((loci+oh, locj), (loci+oh, locj+ow-1))
            ln3 = connect((loci, locj-1), (loci+oh-1, locj-1))
            ln4 = connect((loci, locj+ow), (loci+oh-1, locj+ow))
            if succ > 1:
                fixxer = {
                    (choice(bgcols), choice(totuple(xx))) for xx in [ln1, ln2, ln3, ln4]
                }
                gi = paint(gi, fixxer)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

TWO = 2

F = False

T = True

NEG_UNITY = (-1, -1)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

def leastcommon(
    container: Container
) -> Any:
    """ least common item """
    return min(set(container), key=container.count)

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

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

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

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

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def inbox(
    patch: Patch
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def box(
    patch: Patch
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_890034e9(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = rbind(greater, TWO)
    x1 = chain(x0, minimum, shape)
    x2 = objects(I, T, F, F)
    x3 = sfilter(x2, x1)
    x4 = fork(equality, toindices, box)
    x5 = sfilter(x3, x4)
    x6 = totuple(x5)
    x7 = apply(color, x6)
    x8 = leastcommon(x7)
    x9 = ofcolor(I, x8)
    x10 = inbox(x9)
    x11 = recolor(ZERO, x10)
    x12 = occurrences(I, x11)
    x13 = normalize(x9)
    x14 = shift(x13, NEG_UNITY)
    x15 = lbind(shift, x14)
    x16 = mapply(x15, x12)
    x17 = fill(I, x8, x16)
    return x17


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_890034e9(inp)
        assert pred == _to_grid(expected), f"{name} failed"
