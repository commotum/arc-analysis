"""
# [058] 28e73c20.json
* ex_nihilo
* mimic_pattern
"""

def p(j):A=[i for r in j for i,x in enumerate(r)if x>0];c,E=min(A),max(A)+1;return[r[c:E]*2 for r in j if max(r)>0]