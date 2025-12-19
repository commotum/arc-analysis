"""
# [392] f8c80d96.json
* pattern_expansion
* background_filling
"""

L=len
R=range
def p(g):
 C=max(sum(g,[]))
 g=[[5 if c==0 else c for c in r] for r in g]
 return g