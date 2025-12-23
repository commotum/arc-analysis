# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "6cf79266"
SERIAL = "162"
URL    = "https://arcprize.org/play?task=6cf79266"

# --- Code Golf Concepts ---
CONCEPTS = [
    "rectangle_guessing",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 0],
    [5, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 5],
    [5, 5, 5, 5, 5, 5, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5],
    [0, 5, 0, 5, 0, 5, 0, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 5, 5, 5, 0, 0, 0, 5],
    [0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 0, 5, 0, 0, 0, 5],
    [5, 5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 0, 0, 0, 5],
    [0, 0, 5, 0, 0, 0, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5],
    [5, 5, 5, 0, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5],
    [5, 0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 0, 5, 5, 5, 0, 5],
    [5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5],
    [0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5],
    [5, 0, 0, 5, 0, 5, 0, 0, 0, 5, 5, 5, 0, 5, 0, 5, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 0, 0, 0, 5, 0, 5, 5, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 5],
    [5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 0, 5, 0, 0, 0, 5],
], dtype=int)

E1_OUT = np.array([
    [5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 0],
    [5, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 5],
    [5, 5, 5, 5, 5, 5, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5],
    [0, 5, 0, 5, 0, 5, 0, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 5, 5, 5, 1, 1, 1, 5],
    [0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 5, 0, 5, 1, 1, 1, 5],
    [5, 5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 1, 1, 1, 5],
    [0, 0, 5, 0, 0, 0, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5],
    [5, 5, 5, 0, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5],
    [5, 0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 0, 5, 5, 5, 0, 5],
    [5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5],
    [0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5],
    [5, 0, 0, 5, 0, 5, 0, 0, 0, 5, 5, 5, 0, 5, 0, 5, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 0, 0, 0, 5, 0, 5, 5, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 5],
    [5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 0, 5, 0, 0, 0, 5],
], dtype=int)

E2_IN = np.array([
    [3, 3, 3, 3, 0, 3, 0, 3, 0, 3, 3, 0, 0, 3, 3, 3, 0, 3, 0, 0],
    [0, 0, 3, 3, 0, 0, 3, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 3, 3, 0],
    [3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3, 3],
    [3, 0, 3, 3, 0, 0, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3, 0, 3, 3, 0],
    [0, 0, 0, 3, 0, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0],
    [3, 3, 0, 0, 3, 3, 0, 3, 3, 3, 3, 0, 0, 3, 0, 3, 3, 3, 3, 0],
    [0, 3, 0, 0, 0, 0, 3, 3, 0, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3],
    [0, 3, 3, 0, 0, 0, 0, 3, 0, 3, 3, 0, 3, 3, 0, 0, 3, 3, 3, 3],
    [0, 0, 0, 3, 3, 0, 0, 3, 3, 3, 3, 3, 0, 3, 0, 3, 0, 3, 3, 3],
    [3, 0, 3, 3, 0, 3, 3, 3, 0, 0, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3],
    [3, 0, 0, 3, 0, 0, 0, 3, 3, 3, 3, 0, 0, 3, 0, 3, 0, 3, 3, 3],
    [0, 3, 3, 0, 0, 0, 3, 3, 0, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 3],
    [0, 0, 3, 0, 3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 0, 0, 3, 0, 3, 0],
    [3, 0, 3, 3, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3],
    [0, 0, 3, 0, 3, 3, 0, 0, 3, 0, 3, 0, 3, 3, 0, 3, 3, 3, 0, 0],
    [3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 3],
    [0, 3, 0, 3, 0, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0, 3, 3, 0, 0, 0],
    [3, 0, 0, 3, 0, 3, 3, 0, 3, 0, 0, 3, 0, 0, 3, 3, 3, 3, 3, 3],
    [3, 0, 3, 3, 0, 3, 3, 3, 0, 0, 0, 3, 0, 3, 0, 3, 3, 3, 0, 3],
], dtype=int)

E2_OUT = np.array([
    [3, 3, 3, 3, 0, 3, 0, 3, 0, 3, 3, 0, 0, 3, 3, 3, 0, 3, 0, 0],
    [0, 0, 3, 3, 0, 0, 3, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 3, 3, 0],
    [3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3, 3],
    [3, 0, 3, 3, 0, 0, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3, 0, 3, 3, 0],
    [0, 0, 0, 3, 0, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0],
    [3, 3, 0, 0, 3, 3, 0, 3, 3, 3, 3, 0, 0, 3, 0, 3, 3, 3, 3, 0],
    [0, 3, 0, 1, 1, 1, 3, 3, 0, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0],
    [3, 0, 3, 1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3],
    [0, 3, 3, 1, 1, 1, 0, 3, 0, 3, 3, 0, 3, 3, 0, 0, 3, 3, 3, 3],
    [0, 0, 0, 3, 3, 0, 0, 3, 3, 3, 3, 3, 0, 3, 0, 3, 0, 3, 3, 3],
    [3, 0, 3, 3, 0, 3, 3, 3, 0, 0, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3],
    [3, 0, 0, 3, 0, 0, 0, 3, 3, 3, 3, 0, 0, 3, 0, 3, 0, 3, 3, 3],
    [0, 3, 3, 0, 0, 0, 3, 3, 0, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 3],
    [0, 0, 3, 0, 3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 0, 0, 3, 0, 3, 0],
    [3, 0, 3, 3, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3],
    [0, 0, 3, 0, 3, 3, 0, 0, 3, 0, 3, 0, 3, 3, 0, 3, 3, 3, 0, 0],
    [3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 3],
    [0, 3, 0, 3, 0, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0, 3, 3, 0, 0, 0],
    [3, 0, 0, 3, 0, 3, 3, 0, 3, 0, 0, 3, 0, 0, 3, 3, 3, 3, 3, 3],
    [3, 0, 3, 3, 0, 3, 3, 3, 0, 0, 0, 3, 0, 3, 0, 3, 3, 3, 0, 3],
], dtype=int)

E3_IN = np.array([
    [7, 0, 7, 7, 7, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 0, 7, 7, 7],
    [0, 0, 7, 0, 7, 0, 7, 0, 7, 7, 7, 0, 0, 0, 0, 7, 7, 0, 0, 7],
    [0, 0, 0, 0, 0, 7, 0, 0, 7, 7, 7, 7, 0, 7, 0, 0, 0, 0, 7, 0],
    [7, 0, 7, 0, 7, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7, 0],
    [0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 7, 7, 7, 7, 0, 0, 7, 0, 0, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 0, 7, 7, 0, 7, 7, 0, 0, 0, 7, 0, 7],
    [0, 0, 0, 7, 0, 7, 0, 0, 7, 7, 0, 7, 0, 7, 0, 0, 0, 0, 7, 7],
    [0, 7, 7, 7, 7, 0, 7, 0, 7, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 7, 0, 0, 0, 0, 7, 7, 7, 0, 0, 7, 7, 0, 0, 0, 7, 7],
    [7, 7, 0, 7, 7, 7, 0, 7, 0, 0, 7, 0, 7, 7, 0, 7, 7, 0, 7, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 7, 7, 0],
    [7, 7, 0, 0, 7, 7, 7, 0, 7, 7, 7, 7, 0, 7, 0, 0, 7, 7, 7, 7],
    [0, 7, 0, 7, 7, 7, 0, 0, 0, 7, 7, 0, 7, 7, 0, 7, 0, 0, 7, 7],
    [0, 0, 7, 7, 0, 7, 7, 7, 7, 7, 0, 7, 7, 0, 7, 7, 7, 0, 7, 7],
    [0, 0, 7, 7, 7, 0, 7, 0, 7, 7, 0, 7, 0, 7, 7, 7, 0, 7, 7, 7],
    [7, 0, 7, 7, 7, 0, 7, 0, 7, 7, 7, 7, 7, 0, 0, 7, 7, 7, 0, 0],
    [7, 7, 7, 0, 0, 0, 7, 7, 7, 0, 7, 7, 0, 7, 0, 7, 0, 0, 0, 0],
    [7, 7, 7, 0, 0, 0, 7, 0, 7, 7, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 7, 7, 0, 7, 0, 0, 0, 7, 0, 7, 7, 7, 0, 7],
    [0, 7, 7, 0, 7, 7, 0, 7, 0, 0, 7, 7, 7, 7, 0, 0, 7, 0, 7, 7],
], dtype=int)

E3_OUT = np.array([
    [7, 0, 7, 7, 7, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 0, 7, 7, 7],
    [0, 0, 7, 0, 7, 0, 7, 0, 7, 7, 7, 0, 0, 0, 0, 7, 7, 0, 0, 7],
    [0, 0, 0, 0, 0, 7, 0, 0, 7, 7, 7, 7, 0, 7, 0, 0, 0, 0, 7, 0],
    [7, 0, 7, 0, 7, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7, 0],
    [0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 7, 7, 7, 7, 0, 0, 7, 0, 0, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 0, 7, 7, 0, 7, 7, 1, 1, 1, 7, 0, 7],
    [0, 0, 0, 7, 0, 7, 0, 0, 7, 7, 0, 7, 0, 7, 1, 1, 1, 0, 7, 7],
    [0, 7, 7, 7, 7, 0, 7, 0, 7, 0, 0, 7, 7, 7, 1, 1, 1, 0, 0, 7],
    [0, 0, 0, 7, 0, 0, 0, 0, 7, 7, 7, 0, 0, 7, 7, 0, 0, 0, 7, 7],
    [7, 7, 0, 7, 7, 7, 0, 7, 0, 0, 7, 0, 7, 7, 0, 7, 7, 0, 7, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 7, 7, 0],
    [7, 7, 0, 0, 7, 7, 7, 0, 7, 7, 7, 7, 0, 7, 0, 0, 7, 7, 7, 7],
    [0, 7, 0, 7, 7, 7, 0, 0, 0, 7, 7, 0, 7, 7, 0, 7, 0, 0, 7, 7],
    [0, 0, 7, 7, 0, 7, 7, 7, 7, 7, 0, 7, 7, 0, 7, 7, 7, 0, 7, 7],
    [0, 0, 7, 7, 7, 0, 7, 0, 7, 7, 0, 7, 0, 7, 7, 7, 0, 7, 7, 7],
    [7, 0, 7, 7, 7, 0, 7, 0, 7, 7, 7, 7, 7, 0, 0, 7, 7, 7, 0, 0],
    [7, 7, 7, 1, 1, 1, 7, 7, 7, 0, 7, 7, 0, 7, 0, 7, 0, 0, 0, 0],
    [7, 7, 7, 1, 1, 1, 7, 0, 7, 7, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0],
    [7, 0, 0, 1, 1, 1, 7, 7, 0, 7, 0, 0, 0, 7, 0, 7, 7, 7, 0, 7],
    [0, 7, 7, 0, 7, 7, 0, 7, 0, 0, 7, 7, 7, 7, 0, 0, 7, 0, 7, 7],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 4, 0, 4, 4, 0, 4, 4, 4, 0, 0, 0, 4, 0, 4, 4, 4, 4, 4, 0],
    [0, 0, 4, 4, 0, 0, 4, 0, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0],
    [4, 4, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 0],
    [4, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 4, 0, 0],
    [4, 0, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 0, 4],
    [4, 4, 0, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 0, 4, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4],
    [4, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 0, 0],
    [4, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 4, 0, 4, 4, 4, 0, 4, 0, 4],
    [0, 0, 0, 4, 0, 4, 0, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 4, 4],
    [0, 4, 4, 0, 0, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 4, 4, 0, 0, 0, 4, 4],
    [4, 0, 4, 4, 0, 4, 0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 0, 0],
    [0, 4, 4, 4, 4, 0, 0, 4, 0, 4, 0, 0, 4, 4, 0, 4, 4, 4, 4, 4],
    [4, 0, 0, 4, 4, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0],
    [4, 4, 0, 4, 0, 4, 0, 4, 4, 0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4],
    [4, 0, 0, 0, 0, 4, 4, 0, 4, 4, 0, 4, 0, 4, 0, 0, 0, 4, 4, 4],
    [0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 4, 0, 4, 0, 4, 0, 4, 4],
], dtype=int)

T_OUT = np.array([
    [0, 4, 0, 4, 4, 0, 4, 4, 4, 0, 0, 0, 4, 0, 4, 4, 4, 4, 4, 0],
    [0, 0, 4, 4, 0, 0, 4, 0, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0],
    [4, 4, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 0],
    [4, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 4, 0, 0],
    [4, 0, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 0, 4],
    [4, 4, 0, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 0, 4, 0],
    [0, 0, 0, 4, 1, 1, 1, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4],
    [4, 0, 4, 4, 1, 1, 1, 4, 4, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 1, 1, 1, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 0, 0],
    [4, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 4, 0, 4, 4, 4, 0, 4, 0, 4],
    [0, 0, 0, 4, 0, 4, 0, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 4, 4],
    [0, 4, 4, 0, 0, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 4, 4, 0, 0, 0, 4, 4],
    [4, 0, 4, 4, 0, 4, 0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 0, 0],
    [0, 4, 4, 4, 4, 0, 0, 4, 0, 4, 0, 0, 4, 4, 0, 4, 4, 4, 4, 4],
    [4, 0, 0, 4, 4, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0],
    [4, 4, 0, 4, 0, 4, 0, 4, 4, 0, 0, 4, 4, 4, 1, 1, 1, 0, 4, 4],
    [4, 0, 1, 1, 1, 4, 4, 0, 4, 4, 0, 4, 0, 4, 1, 1, 1, 4, 4, 4],
    [0, 0, 1, 1, 1, 4, 4, 4, 4, 0, 4, 0, 0, 4, 1, 1, 1, 0, 0, 0],
    [4, 4, 1, 1, 1, 0, 0, 4, 4, 0, 0, 0, 4, 0, 4, 0, 4, 0, 4, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range(18)):
 for c in A:
  E,k,W=j[c:c+3]
  for l in A:
   J=l+3
   if sum(E[l:J]+k[l:J]+W[l:J])==0:E[l:J]=k[l:J]=W[l:J]=[1]*3
 return j


# --- Code Golf Solution (Compressed) ---
def q(g, k=0):
    return eval('1,1,1'.join(re.split(('(.{55})0, 0, 0' * 3)[7:], str(k or p(g, g)))))


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

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

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

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

def generate_6cf79266(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    nfgcs = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(cols, nfgcs)
    gi = canvas(-1, (h, w))
    fgcobj = {(choice(ccols), ij) for ij in asindices(gi)}
    gi = paint(gi, fgcobj)
    num = unifint(diff_lb, diff_ub, (int(0.25 * h * w), int(0.6 * h * w)))
    inds = asindices(gi)
    locs = sample(totuple(inds), num)
    gi = fill(gi, 0, locs)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    cands = asindices(canvas(-1, (h - 2, w - 2)))
    locs = sample(totuple(cands), noccs)
    mini = asindices(canvas(-1, (3, 3)))
    for ij in locs:
        gi = fill(gi, 0, shift(mini, ij))
    trg = recolor(0, mini)
    occs = occurrences(gi, trg)
    go = tuple(e for e in gi)
    for occ in occs:
        go = fill(go, 1, shift(mini, occ))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ContainerContainer = Container[Container]

ZERO = 0

ONE = 1

THREE_BY_THREE = (3, 3)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_6cf79266(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = canvas(ZERO, THREE_BY_THREE)
    x1 = asobject(x0)
    x2 = occurrences(I, x1)
    x3 = lbind(shift, x1)
    x4 = mapply(x3, x2)
    x5 = fill(I, ONE, x4)
    return x5


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_6cf79266(inp)
        assert pred == _to_grid(expected), f"{name} failed"
