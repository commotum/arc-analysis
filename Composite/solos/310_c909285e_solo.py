# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "c909285e"
SERIAL = "310"
URL    = "https://arcprize.org/play?task=c909285e"

# --- Code Golf Concepts ---
CONCEPTS = [
    "find_the_intruder",
    "crop",
    "rectangle_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 2, 4, 8, 5, 0, 4, 2, 8, 0, 5, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [0, 0, 2, 4, 8, 5, 0, 4, 2, 8, 0, 5, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [2, 2, 2, 4, 2, 5, 2, 4, 2, 2, 2, 5, 2, 2, 2, 4, 2, 5, 2, 4, 2, 2, 2, 5],
    [4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5],
    [8, 8, 2, 4, 8, 5, 8, 4, 2, 8, 8, 5, 8, 8, 2, 4, 8, 5, 8, 4, 2, 8, 8, 5],
    [5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 2, 4, 8, 3, 0, 4, 2, 8, 0, 3, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5],
    [2, 2, 2, 4, 2, 3, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4, 2, 5, 2, 4, 2, 2, 2, 5],
    [8, 8, 2, 4, 8, 3, 8, 4, 2, 8, 8, 3, 8, 8, 2, 4, 8, 5, 8, 4, 2, 8, 8, 5],
    [0, 0, 2, 4, 8, 3, 0, 4, 2, 8, 0, 3, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 2, 4, 8, 5, 0, 4, 2, 8, 0, 5, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [0, 0, 2, 4, 8, 5, 0, 4, 2, 8, 0, 5, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [2, 2, 2, 4, 2, 5, 2, 4, 2, 2, 2, 5, 2, 2, 2, 4, 2, 5, 2, 4, 2, 2, 2, 5],
    [4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5],
    [0, 0, 2, 4, 8, 5, 0, 4, 2, 8, 0, 5, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 2, 4, 8, 5, 0, 4, 2, 8, 0, 5, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5],
    [2, 2, 2, 4, 2, 5, 2, 4, 2, 2, 2, 5, 2, 2, 2, 4, 2, 5, 2, 4, 2, 2, 2, 5],
    [0, 0, 2, 4, 8, 5, 0, 4, 2, 8, 0, 5, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [0, 0, 2, 4, 8, 5, 0, 4, 2, 8, 0, 5, 0, 0, 2, 4, 0, 5, 0, 4, 2, 0, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

E1_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3],
    [3, 0, 4, 2, 8, 0, 3],
    [3, 4, 4, 4, 4, 4, 3],
    [3, 2, 4, 2, 2, 2, 3],
    [3, 8, 4, 2, 8, 8, 3],
    [3, 0, 4, 2, 8, 0, 3],
    [3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E2_IN = np.array([
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [3, 3, 2, 3, 3, 8, 3, 3, 2, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3],
    [1, 1, 2, 3, 1, 8, 1, 3, 2, 1, 1, 8, 1, 1, 8, 3, 1, 8, 1, 3, 8, 1, 1, 8, 1, 1],
    [8, 8, 2, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 2, 3, 1, 8, 0, 3, 2, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [3, 3, 2, 3, 3, 8, 3, 3, 2, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3],
    [8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 8, 3, 1, 8, 1, 3, 8, 1, 1, 8, 1, 1, 8, 3, 1, 8, 1, 3, 8, 1, 1, 8, 1, 1],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 8, 3, 1, 8, 1, 3, 8, 1, 1, 8, 1, 1, 8, 3, 1, 8, 1, 3, 8, 1, 1, 8, 1, 1],
    [0, 0, 8, 3, 1, 8, 0, 3, 8, 1, 0, 8, 0, 0, 8, 3, 0, 8, 0, 3, 8, 0, 0, 8, 1, 0],
], dtype=int)

E2_OUT = np.array([
    [2, 2, 2, 2, 2, 2, 2],
    [2, 3, 3, 8, 3, 3, 2],
    [2, 3, 1, 8, 1, 3, 2],
    [2, 8, 8, 8, 8, 8, 2],
    [2, 3, 1, 8, 0, 3, 2],
    [2, 3, 3, 8, 3, 3, 2],
    [2, 2, 2, 2, 2, 2, 2],
], dtype=int)

E3_IN = np.array([
    [0, 0, 3, 1, 8, 5, 0, 1, 3, 8, 0, 5, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [0, 0, 3, 1, 8, 5, 0, 1, 3, 8, 0, 5, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3],
    [1, 1, 3, 1, 8, 5, 1, 1, 3, 8, 1, 5, 1, 1, 3, 1, 1, 5, 1, 8, 3, 1, 1, 5, 8, 1, 3, 1],
    [8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 3, 1, 8, 5, 0, 1, 3, 8, 0, 5, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [1, 1, 3, 1, 8, 5, 1, 1, 3, 8, 1, 5, 1, 1, 3, 1, 1, 5, 1, 8, 3, 1, 1, 5, 8, 1, 3, 1],
    [3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3],
    [8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8],
    [0, 0, 3, 1, 8, 5, 0, 1, 3, 8, 0, 5, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 3, 1, 8, 5, 0, 1, 3, 8, 0, 5, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [0, 0, 3, 1, 8, 5, 0, 1, 3, 8, 0, 5, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3],
    [1, 1, 3, 1, 8, 5, 1, 1, 3, 8, 1, 5, 1, 1, 3, 1, 1, 5, 1, 8, 3, 1, 1, 5, 8, 1, 3, 1],
    [0, 0, 3, 1, 8, 5, 0, 1, 3, 8, 0, 5, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 3, 1, 8, 6, 0, 1, 3, 8, 0, 6, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [8, 8, 3, 8, 8, 6, 8, 8, 3, 8, 8, 6, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8],
    [3, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3],
    [0, 0, 3, 1, 8, 6, 0, 1, 3, 8, 0, 6, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [0, 0, 3, 1, 8, 6, 0, 1, 3, 8, 0, 6, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8, 8, 5, 8, 8, 3, 8],
    [0, 0, 3, 1, 8, 5, 0, 1, 3, 8, 0, 5, 0, 0, 3, 1, 0, 5, 0, 8, 3, 0, 0, 5, 8, 0, 3, 1],
    [3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3],
    [1, 1, 3, 1, 8, 5, 1, 1, 3, 8, 1, 5, 1, 1, 3, 1, 1, 5, 1, 8, 3, 1, 1, 5, 8, 1, 3, 1],
], dtype=int)

E3_OUT = np.array([
    [6, 6, 6, 6, 6, 6, 6],
    [6, 0, 1, 3, 8, 0, 6],
    [6, 8, 8, 3, 8, 8, 6],
    [6, 3, 3, 3, 3, 3, 6],
    [6, 0, 1, 3, 8, 0, 6],
    [6, 0, 1, 3, 8, 0, 6],
    [6, 6, 6, 6, 6, 6, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 3, 2, 0, 4, 0, 3, 1, 0, 0, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 3, 2, 0, 4, 0, 3, 1, 0, 0, 4],
    [1, 1, 1, 2, 3, 4, 1, 2, 1, 3, 1, 4, 1, 1, 3, 2, 1, 4, 1, 3, 1, 1, 1, 4],
    [2, 2, 2, 2, 3, 4, 2, 2, 2, 3, 2, 4, 2, 2, 3, 2, 2, 4, 2, 3, 2, 2, 2, 4],
    [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 3, 2, 0, 4, 0, 3, 1, 0, 0, 4],
    [2, 2, 2, 2, 3, 4, 2, 2, 2, 3, 2, 4, 2, 2, 3, 2, 2, 4, 2, 3, 2, 2, 2, 4],
    [1, 1, 1, 2, 3, 4, 1, 2, 1, 3, 1, 4, 1, 1, 3, 2, 1, 4, 1, 3, 1, 1, 1, 4],
    [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 3, 2, 0, 4, 0, 3, 1, 0, 0, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 3, 2, 0, 4, 0, 3, 1, 0, 0, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 3, 2, 0, 4, 0, 3, 1, 0, 0, 4],
    [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3, 8, 8, 8, 8, 8, 8, 3, 3, 3, 4],
    [2, 2, 2, 2, 3, 4, 2, 2, 2, 3, 2, 4, 2, 2, 8, 2, 2, 4, 2, 8, 2, 2, 2, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 8, 2, 0, 4, 0, 8, 1, 0, 0, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 8, 4, 4, 4, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 8, 2, 0, 4, 0, 8, 1, 0, 0, 4],
    [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3, 8, 8, 8, 8, 8, 8, 3, 3, 3, 4],
    [1, 1, 1, 2, 3, 4, 1, 2, 1, 3, 1, 4, 1, 1, 3, 2, 1, 4, 1, 3, 1, 1, 1, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 3, 2, 0, 4, 0, 3, 1, 0, 0, 4],
    [0, 0, 1, 2, 3, 4, 0, 2, 1, 3, 0, 4, 0, 0, 3, 2, 0, 4, 0, 3, 1, 0, 0, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
], dtype=int)

T_OUT = np.array([
    [8, 8, 8, 8, 8, 8],
    [8, 2, 2, 4, 2, 8],
    [8, 2, 0, 4, 0, 8],
    [8, 4, 4, 4, 4, 8],
    [8, 2, 0, 4, 0, 8],
    [8, 8, 8, 8, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
from collections import Counter
def p(m):
 c=Counter(e for r in m for e in r if e).most_common()
 if not c:return[]
 l=c[-1][0];O=p=-1
 for i,r in enumerate(m):
  if l in r:
   if O<0:O=i
   p=i
 S=U=-1
 for i in range(len(m[0])):
  if any(m[j][i]==l for j in range(O,p+1)):
   if S<0:S=i
   U=i
 return[r[S:U+1]for r in m[O:p+1]]


# --- Code Golf Solution (Compressed) ---
def q(a, *n):
    return [b for b in zip(*(n or p(a, *a))) if {*b} - ({*a[1]} & {*a[12]})]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

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

def generate_c909285e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nfronts = unifint(diff_lb, diff_ub, (1, (h + w) // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    boxcol = choice(remcols)
    remcols = remove(boxcol, remcols)
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    for k in range(nfronts):
        ff = choice((hfrontier, vfrontier))
        loc = choice(inds)
        inds = remove(loc, inds)
        col = choice(remcols)
        gi = fill(gi, col, ff(loc))
    oh = unifint(diff_lb, diff_ub, (3, max(3, (h - 2) // 2)))
    ow = unifint(diff_lb, diff_ub, (3, max(3, (w - 2) // 2)))
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, boxcol, bx)
    go = subgrid(bx, gi)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ONE = 1

def multiply(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

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

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_c909285e(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = lbind(contained, ONE)
    x2 = chain(flip, x1, shape)
    x3 = sfilter(x0, x2)
    x4 = fork(equality, toindices, box)
    x5 = sfilter(x3, x4)
    x6 = fork(multiply, height, width)
    x7 = argmin(x5, x6)
    x8 = subgrid(x7, I)
    return x8


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_c909285e(inp)
        assert pred == _to_grid(expected), f"{name} failed"
