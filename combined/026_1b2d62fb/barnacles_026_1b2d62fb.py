"""
# [027-R] 1b60fb0c.json
* pattern_deconstruction
* pattern_rotation
* pattern_expansion
"""

p=lambda j:[[8*(not A|B)for(A,B)in zip(A,A[4:])]for A in j]