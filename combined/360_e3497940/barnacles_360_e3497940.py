"""
# [361] e40b9e2f.json
* pattern_expansion
* pattern_reflection
* pattern_rotation
"""

p=lambda g:[[g[i][j]or g[i][8-j]if g[i][j]*g[i][8-j]==0 else g[i][j]for j in range(4)]for i in range(len(g))]