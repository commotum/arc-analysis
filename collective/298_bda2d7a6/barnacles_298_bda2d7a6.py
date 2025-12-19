"""
# [298] bda2d7a6.json
* recoloring
* pairwise_analogy
* pattern_modification
* color_permutation
"""

def p(j):A=len(j)//2;c=[j[i][i]for i in range(A)];E={c[i]:c[i-1]for i in range(A)};return[[E[i]for i in r]for r in j]