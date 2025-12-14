"""
# [293] ba97ae07.json
* pattern_modification
* pairwise_analogy
* rettangle_guessing
* recoloring
"""

def p(j):
 for A in j:A[::3]=[6 if v==4 else v for v in A[::3]]
 return j