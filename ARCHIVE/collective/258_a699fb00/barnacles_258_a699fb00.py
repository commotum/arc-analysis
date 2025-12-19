"""
# [258] a699fb00.json
* pattern_expansion
* connect_the_dots
"""

def p(j):
 for A in j:
  for c in range(len(A)-2):
   if A[c]&A[c+2]:A[c+1]=2
 return j