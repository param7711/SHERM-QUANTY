"""Diff two cell-run logs, ignoring only lines that legitimately vary
(wall-clock timings). Anything else differing means non-determinism."""
import re, sys
VOL = [
    re.compile(r'\d+\.\d+s'),                 # any "12.3s" timing
    re.compile(r'secs'),
    re.compile(r'RUNTIME \d'),
    re.compile(r'wall clock'),
    re.compile(r'projected / actual'),
    re.compile(r'in \d+\.\d+s'),
]
def load(p):
    out = []
    for i, l in enumerate(open(p)):
        l = l.rstrip('\n')
        if any(v.search(l) for v in VOL):
            continue
        out.append(l)
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
if a == b:
    print(f'DETERMINISM OK: {len(a)} comparable lines IDENTICAL between runs.')
    sys.exit(0)
print(f'DIFFERENCES: {len(a)} vs {len(b)} comparable lines')
import difflib
n = 0
for d in difflib.unified_diff(a, b, 'run1', 'run2', lineterm='', n=1):
    print(d); n += 1
    if n > 120: print('... truncated'); break
sys.exit(1)
