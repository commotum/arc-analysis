"""
# [292] ba26e723.json
* pattern_modification
* pairwise_analogy
* recoloring
"""

def p(j):
 for A in j:A[::3]=[6 if v==4 else v for v in A[::3]]
 return j