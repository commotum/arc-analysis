"""
# [313] caa06a1f.json
* pattern_expansion
* image_filling
"""

def p(j):
 for A in j:
  for c in A:
   if c and c-5:A[:]=[c*(x==5)+x*(x!=5)for x in A];break
 return j