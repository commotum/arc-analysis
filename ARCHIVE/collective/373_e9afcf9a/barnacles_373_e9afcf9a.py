"""
# [373] e9afcf9a.json
* pattern_modification
"""

p=lambda g:[[[g[i][j],g[1-i][j]][j%2]for j in range(6)]for i in range(2)]