"""
# [315] cce03e0d.json
* image_repetition
* image_expansion
* pairwise_analogy
"""

p=lambda j,A=range(9):[[j[r%3][c%3]*(j[r//3][c//3]==2)for c in A]for r in A]