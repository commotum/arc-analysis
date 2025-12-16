"""
# [042] 22233c11.json
* pattern_expansion
* size_guessing
"""

def p(j,A=0):
 for c in j:
  for E,k in enumerate(c):
   if k:A=(not A)*k
   else:c[E]=A
 return j