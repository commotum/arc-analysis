# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a64e4611"
SERIAL = "255"
URL    = "https://arcprize.org/play?task=a64e4611"

# --- Code Golf Concepts ---
CONCEPTS = [
    "background_filling",
    "rectangle_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [8, 8, 0, 8, 0, 8, 0, 8, 8, 8, 8, 8, 0, 8, 8, 8, 0, 8, 0, 0, 8, 0, 8, 0, 0, 0, 8, 8, 0, 8],
    [0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 8, 0, 0, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 8, 8, 0, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8, 0, 8, 0, 8, 8, 8, 8, 0, 0, 8, 0, 0, 0],
    [0, 8, 8, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 8, 8, 0, 8, 8, 0, 0, 0, 8, 8, 0, 8, 0, 0, 0, 0, 0],
    [8, 8, 8, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 8, 8, 0, 0, 8],
    [0, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 8, 0, 8, 0, 0, 0, 8],
    [0, 8, 8, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 8, 0, 8, 8, 8],
    [0, 8, 8, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8, 8, 8, 0, 0, 8, 8],
    [8, 0, 8, 8, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8],
    [8, 8, 8, 0, 8, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 8, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8],
    [8, 0, 8, 8, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 8],
    [8, 0, 8, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 8, 0, 0, 0, 8, 0, 8, 8],
    [0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 8, 8, 0, 8, 8, 0, 0, 8],
    [8, 0, 8, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 8, 8, 0, 0, 0, 8, 8, 0],
    [8, 0, 8, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 8, 0, 8, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 8, 8, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 8, 0, 8, 8, 8],
    [8, 8, 0, 0, 8, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 8, 0, 0, 0, 0, 8, 8, 8, 8],
    [0, 8, 8, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 8, 8, 0, 0, 0, 8, 8],
    [0, 8, 8, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 8],
    [8, 0, 8, 8, 8, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 8, 0, 8, 8, 0, 0, 0, 8],
    [8, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8, 0, 0, 8, 0],
    [0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 8, 8, 0, 8, 8, 8, 0, 0, 0],
    [8, 8, 8, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 8, 0, 8, 8, 0, 8, 0, 8, 8, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 0, 8, 0, 8, 0, 8, 8, 8, 8, 8, 0, 8, 8, 8, 0, 8, 0, 0, 8, 0, 8, 0, 0, 0, 8, 8, 0, 8],
    [0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 8, 0, 0, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 8, 8, 0, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8, 0, 8, 0, 8, 8, 8, 8, 0, 0, 8, 0, 0, 0],
    [0, 8, 8, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 8, 8, 0, 8, 8, 0, 0, 0, 8, 8, 0, 8, 0, 0, 0, 0, 0],
    [8, 8, 8, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 8, 8, 0, 0, 8],
    [0, 8, 0, 0, 0, 8, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 8, 8, 8, 8, 0, 8, 0, 8, 0, 0, 0, 8],
    [0, 8, 8, 8, 8, 0, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 8, 0, 0, 0, 0, 0, 8, 0, 8, 8, 8],
    [0, 8, 8, 8, 8, 0, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 8, 0, 8, 0, 8, 8, 8, 0, 0, 8, 8],
    [8, 0, 8, 8, 0, 8, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 0, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8],
    [8, 8, 8, 0, 8, 8, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 8, 0, 0, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 8, 8, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 8, 0, 0, 8, 0, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8],
    [8, 0, 8, 8, 0, 0, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 8],
    [8, 0, 8, 0, 0, 8, 8, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 8, 0, 8, 8, 0, 0, 0, 8, 0, 8, 8],
    [0, 0, 8, 8, 8, 8, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 0, 0, 0, 8, 8, 0, 8, 8, 0, 0, 8],
    [8, 0, 8, 0, 0, 8, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 8, 8, 0, 8, 8, 0, 0, 0, 8, 8, 0],
    [8, 0, 8, 8, 0, 8, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 8, 8, 8, 0, 8, 0, 8, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 8, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 8, 0, 0, 0, 0, 8, 0, 8, 8, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 8, 0, 0, 0, 0, 8, 0, 8, 0, 8, 8, 8],
    [8, 8, 0, 0, 8, 8, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 8, 0, 8, 0, 0, 0, 0, 8, 8, 8, 8],
    [0, 8, 8, 8, 8, 0, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 0, 8, 0, 8, 0, 8, 8, 0, 0, 0, 8, 8],
    [0, 8, 8, 0, 8, 8, 8, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 8],
    [8, 0, 8, 8, 8, 0, 8, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 8, 8, 8, 0, 8, 0, 8, 8, 0, 0, 0, 8],
    [8, 0, 8, 0, 8, 0, 8, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 0, 8, 8, 0, 8, 8, 0, 8, 0, 0, 8, 0],
    [0, 8, 8, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 8, 0, 0, 8, 8, 0, 8, 8, 8, 0, 0, 0],
    [8, 8, 8, 0, 8, 0, 0, 8, 0, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 0, 8, 0, 8, 8, 0, 8, 0, 8, 8, 0],
], dtype=int)

E2_IN = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
], dtype=int)

E2_OUT = np.array([
    [1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [1, 0, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [1, 1, 0, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
], dtype=int)

E3_IN = np.array([
    [0, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2],
    [0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 2, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 2],
    [0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2],
    [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0],
    [2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2, 2],
    [2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0, 2, 0],
    [2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [0, 2, 0, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 2, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 2, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2],
    [0, 0, 0, 2, 0, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 0, 2, 2, 2, 0, 0, 0, 0, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 0, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 2],
    [0, 0, 2, 2, 0, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2],
    [2, 0, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [2, 2, 0, 0, 2, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0],
    [2, 0, 0, 0, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2, 2],
    [2, 0, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0, 2, 0],
    [2, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 4, 4, 0, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 4, 0, 4, 4, 4, 0, 0, 0],
    [4, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 4],
    [0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 4],
    [4, 0, 0, 0, 4, 4, 4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4],
    [4, 0, 4, 4, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 4, 0, 4, 0],
    [0, 0, 4, 0, 4, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    [4, 0, 4, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0, 0, 4, 0, 0, 4, 4, 0, 4, 4],
    [0, 4, 0, 4, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4, 0, 0, 4, 4],
    [4, 4, 0, 4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 4, 0, 0, 4, 4, 4, 0, 4, 0, 0],
    [0, 0, 4, 0, 4, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0, 4, 4, 0, 4],
    [4, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 4, 0, 4, 0, 4, 0, 0, 4, 4],
    [0, 4, 4, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 4, 0, 4, 0, 0, 0, 4, 0, 0, 4, 4, 4, 4],
    [4, 4, 0, 0, 0, 0, 4, 4, 0, 4, 0, 0, 0, 0, 4, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 4, 4, 0],
    [4, 0, 4, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0, 0, 4, 0, 4, 4, 0, 4, 0, 4],
    [0, 4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 4, 4, 4, 0, 0, 4, 4, 0, 0, 0, 4, 0],
    [0, 0, 4, 0, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 4],
    [4, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 4, 4, 4, 0, 4, 4],
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 4, 4, 0, 4, 0, 4, 4, 0, 0, 0, 3, 3, 0, 4, 4, 4, 4, 4, 0, 3, 0, 4, 0, 4, 4, 4, 0, 0, 0],
    [4, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 3, 3, 0, 0, 4, 4, 0, 4, 0, 3, 0, 0, 0, 0, 4, 4, 0, 4, 4],
    [0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 0, 3, 3, 0, 0, 4, 0, 4, 4, 0, 3, 0, 4, 4, 0, 0, 4, 0, 0, 4],
    [4, 0, 0, 0, 4, 4, 4, 0, 4, 0, 0, 3, 3, 0, 4, 0, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 4],
    [4, 0, 4, 4, 4, 0, 4, 0, 0, 4, 0, 3, 3, 0, 0, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 4, 4, 0, 4, 0],
    [0, 0, 4, 0, 4, 0, 0, 0, 4, 4, 0, 3, 3, 0, 4, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    [4, 0, 4, 4, 0, 0, 4, 0, 0, 4, 0, 3, 3, 0, 4, 0, 4, 4, 0, 0, 3, 0, 4, 0, 0, 4, 4, 0, 4, 4],
    [0, 4, 0, 4, 4, 4, 0, 4, 0, 0, 0, 3, 3, 0, 0, 0, 0, 4, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 0, 4, 4, 4, 0, 0, 3, 3, 0, 4, 4, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0, 4, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 0, 3, 3, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4, 0, 0, 4, 4],
    [4, 4, 0, 4, 4, 0, 0, 4, 0, 0, 0, 3, 3, 0, 0, 4, 0, 4, 0, 0, 4, 0, 0, 4, 4, 4, 0, 4, 0, 0],
    [0, 0, 4, 0, 4, 4, 4, 0, 4, 0, 0, 3, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 3, 3, 0, 4, 4, 4, 0, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 3, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 3, 3, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0, 4, 4, 0, 4],
    [4, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 3, 3, 0, 0, 0, 4, 4, 4, 0, 0, 4, 0, 4, 0, 4, 0, 0, 4, 4],
    [0, 4, 4, 4, 4, 0, 4, 0, 0, 0, 0, 3, 3, 0, 4, 0, 4, 4, 0, 4, 0, 0, 0, 4, 0, 0, 4, 4, 4, 4],
    [4, 4, 0, 0, 0, 0, 4, 4, 0, 4, 0, 3, 3, 0, 4, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 4, 4, 0],
    [4, 0, 4, 0, 0, 0, 4, 0, 0, 4, 0, 3, 3, 0, 4, 0, 4, 4, 0, 0, 0, 0, 4, 0, 4, 4, 0, 4, 0, 4],
    [0, 4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 3, 3, 0, 4, 0, 4, 0, 4, 4, 4, 0, 0, 4, 4, 0, 0, 0, 4, 0],
    [0, 0, 4, 0, 4, 0, 4, 4, 0, 0, 0, 3, 3, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 4],
    [4, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 4, 4, 4, 0, 4, 4],
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 3, 3, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
R=range
L=len
P=lambda m:list(map(list,zip(*m[::-1])))
def p(g):
 C=max(sum(g,[]))
 for i in R(4):
  g=P(g)
  h,w=L(g),L(g[0])
  for r in R(h-1):
   for c in R(w-1):
    if g[r][c]==C:
     for y,x in [[0,0],[0,1],[1,0],[1,1]]:
      if g[r+y][c+x]==0:g[r+y][c+x]=10
 for i in R(4):
  g=P(g)   
  for r in R(h):
   M=sorted(set(g[r]))
   if M==[0] or M==[0,3]:
    g[r]=[3]*L(g[r])
 for i in R(4):
  g=P(g)    
  for r in R(h):
   if C not in g[r][:10] and 10 not in g[r][:10]:
    for c in R(w):
     if g[r][c]<1:g[r][c]=3
     else:break
 g=[[0 if c>9 else c for c in r] for r in g]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g, k=9):
    return eval(re.sub(*['\\((%s|(0, )+3.{5}3,)' % re.search('([ ,03]{61,})(.*\\1){3}|$', (g := f'{(*zip(*(~k * g or p(g, k - 1))),)}'))[1], '(?=0, [^0])', '(*[3]*len([\\1]),', '1<'][k < 2::2], g))[::-1]


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

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

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

def generate_a64e4611(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (18, 30))
    w = unifint(diff_lb, diff_ub, (18, 30))
    bgc, noisec = sample(cols, 2)
    lb = int(0.4 * h * w)
    ub = int(0.5 * h * w)
    nbgc = unifint(diff_lb, diff_ub, (lb, ub))
    gi = canvas(noisec, (h, w))
    inds = totuple(asindices(gi))
    bgcinds = sample(inds, nbgc)
    gi = fill(gi, bgc, bgcinds)
    sinds = asindices(canvas(-1, (3, 3)))
    bgcf = recolor(bgc, sinds)
    noisecf = recolor(noisec, sinds)
    addn = set()
    addb = set()
    for occ in occurrences(gi, bgcf):
        occi, occj = occ
        addn.add((randint(0, 2) + occi, randint(0, 2) + occj))
    for occ in occurrences(gi, noisecf):
        occi, occj = occ
        addb.add((randint(0, 2) + occi, randint(0, 2) + occj))
    gi = fill(gi, noisec, addn)
    gi = fill(gi, bgc, addb)
    go = tuple(e for e in gi)
    dim = randint(randint(3, 8), 8)
    locj = randint(3, h - dim - 4)
    spi = choice((0, randint(3, h//2)))
    for j in range(locj, locj + dim):
        ln = connect((spi, j), (h - 1, j))
        gi = fill(gi, bgc, ln)
        go = fill(go, bgc, ln)
    for j in range(locj + 1, locj + dim - 1):
        ln = connect((spi + 1 if spi > 0 else spi, j), (h - 1, j))
        go = fill(go, 3, ln)
    sgns = choice(((-1,), (1,), (-1, 1)))
    startloc = choice((spi, randint(spi + 3, h - 6)))
    hh = randint(3, min(8, h - startloc - 3))
    for sgn in sgns:
        for ii in range(startloc, startloc + hh, 1):
            ln = shoot((ii, locj), (0, sgn))
            gi = fill(gi, bgc, ln)
            go = fill(go, bgc, ln - ofcolor(go, 3))
    for sgn in sgns:
        for ii in range(startloc+1 if startloc > 0 else startloc, startloc + hh - 1, 1):
            ln = shoot((ii, locj+dim-2 if sgn == -1 else locj+1), (0, sgn))
            go = fill(go, 3, ln)
    if len(sgns) == 1 and unifint(diff_lb, diff_ub, (0, 1)) == 1:
        sgns = (-sgns[0],)
        startloc = choice((spi, randint(spi + 3, h - 6)))
        hh = randint(3, min(8, h - startloc - 3))
        for sgn in sgns:
            for ii in range(startloc, startloc + hh, 1):
                ln = shoot((ii, locj), (0, sgn))
                gi = fill(gi, bgc, ln)
                go = fill(go, bgc, ln - ofcolor(go, 3))
        for sgn in sgns:
            for ii in range(startloc+1 if startloc > 0 else startloc, startloc + hh - 1, 1):
                ln = shoot((ii, locj+dim-2 if sgn == -1 else locj+1), (0, sgn))
                go = fill(go, 3, ln)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ONE = 1

TWO = 2

THREE = 3

FOUR = 4

SIX = 6

UNITY = (1, 1)

DOWN = (1, 0)

RIGHT = (0, 1)

ZERO_BY_TWO = (0, 2)

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

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def lowermost(
    patch: Patch
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))

def rightmost(
    patch: Patch
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def dmirror(
    piece: Piece
) -> Piece:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

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

def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])

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

def verify_a64e4611(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = shape(I)
    x2 = add(TWO, x1)
    x3 = canvas(x0, x2)
    x4 = asobject(I)
    x5 = shift(x4, UNITY)
    x6 = paint(x3, x5)
    x7 = double(SIX)
    x8 = astuple(ONE, x7)
    x9 = connect(UNITY, x8)
    x10 = outbox(x9)
    x11 = backdrop(x10)
    x12 = recolor(x0, x11)
    x13 = recolor(THREE, x9)
    x14 = lbind(shift, x13)
    x15 = lbind(mapply, x14)
    x16 = rbind(occurrences, x12)
    x17 = compose(x15, x16)
    x18 = fork(paint, identity, x17)
    x19 = x18(x6)
    x20 = ofcolor(x19, THREE)
    x21 = dmirror(x6)
    x22 = x18(x21)
    x23 = dmirror(x22)
    x24 = ofcolor(x23, THREE)
    x25 = combine(x20, x24)
    x26 = fill(x6, THREE, x25)
    x27 = astuple(TWO, ONE)
    x28 = dneighbors(UNITY)
    x29 = remove(x27, x28)
    x30 = recolor(x0, x29)
    x31 = initset(UNITY)
    x32 = recolor(THREE, x31)
    x33 = combine(x30, x32)
    x34 = recolor(x0, x33)
    x35 = astuple(ONE, THREE)
    x36 = initset(x35)
    x37 = insert(ZERO_BY_TWO, x36)
    x38 = insert(RIGHT, x37)
    x39 = insert(DOWN, x38)
    x40 = recolor(x0, x39)
    x41 = astuple(ONE, TWO)
    x42 = initset(x41)
    x43 = insert(UNITY, x42)
    x44 = recolor(THREE, x43)
    x45 = combine(x40, x44)
    x46 = recolor(x0, x45)
    x47 = lbind(shift, x34)
    x48 = lbind(mapply, x47)
    x49 = rbind(occurrences, x33)
    x50 = compose(x48, x49)
    x51 = fork(paint, identity, x50)
    x52 = lbind(shift, x46)
    x53 = lbind(mapply, x52)
    x54 = rbind(occurrences, x45)
    x55 = compose(x53, x54)
    x56 = fork(paint, identity, x55)
    x57 = compose(x51, x56)
    x58 = compose(rot90, x57)
    x59 = power(x58, FOUR)
    x60 = power(x59, TWO)
    x61 = asindices(x26)
    x62 = box(x61)
    x63 = fill(x26, THREE, x62)
    x64 = x60(x63)
    x65 = trim(x64)
    return x65


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a64e4611(inp)
        assert pred == _to_grid(expected), f"{name} failed"
