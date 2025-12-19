"""
# [312] c9f8e694.json
* recoloring
* pattern_repetition
* color_palette
"""

def p(j):
 for A in j:
  for c in A:
   if c and c-5:A[:]=[c*(x==5)+x*(x!=5)for x in A];break
 return j