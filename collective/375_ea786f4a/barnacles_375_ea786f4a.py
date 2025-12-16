"""
# [376] eb281b96.json
* image_repetition
* image_reflection
"""

def p(j):
 for A in range(len(j)):j[A][A]=j[-A-1][A]=0
 return j