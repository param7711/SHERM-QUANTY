"""Diff two cell-run logs, ignoring ONLY lines that legitimately vary
(wall-clock timings). Anything else differing means non-determinism.

The earlier version stripped timings written as `12.3s` only. This build also
prints INTEGER seconds (`91s`, `409s`) and a bare trailing `secs` COLUMN in the
grid table (`... ctx 288       85.7`), so both leaked through and two identical
runs were reported as differing. Every wall-clock form is normalised here; the
statistics themselves are compared byte-for-byte.
"""
import re, sys, difflib

# A trailing bare float on a grid-table row is the `secs` column. Two row
# shapes exist: intraday_fx's rows START with the run id, fx_sensitivity's
# start with the arm (`ctx12 EUR/USD@1h ...`). Cover both -- matching only the
# first shape let the sensitivity sweep's timings look like non-determinism.
GRID_ROW = re.compile(r'^((?:ctx\d+\s+)?\S+@\d+h\s+.*?)\s+\d+\.\d+\s*$')
SUBS = [
    (re.compile(r'\d+\.\d+\s*s\b'), '<T>'),        # 12.3s
    (re.compile(r'\b\d+s\b'), '<T>'),              # 91s / 409s
    (re.compile(r'RUNTIME\s+\d+\.?\d*'), 'RUNTIME <T>'),
    (re.compile(r'\d+\.\d+\s*min'), '<T> min'),
    (re.compile(r'in \d+\.\d+'), 'in <T>'),
    (re.compile(r'projected\s+\d+'), 'projected <T>'),
]


def norm(line):
    line = line.rstrip('\n')
    m = GRID_ROW.match(line)
    if m:                       # drop the secs column, keep every statistic
        line = m.group(1) + ' <T>'
    for rx, rep in SUBS:
        line = rx.sub(rep, line)
    return re.sub(r'[ \t]+', ' ', line).rstrip()


a = [norm(l) for l in open(sys.argv[1])]
b = [norm(l) for l in open(sys.argv[2])]
if a == b:
    print(f'DETERMINISM OK: {len(a)} lines IDENTICAL after normalising '
          f'wall-clock timings only. Every statistic, grade and verdict '
          f'matches bit-for-bit.')
    sys.exit(0)
print(f'REAL DIFFERENCES: {len(a)} vs {len(b)} lines')
n = 0
for d in difflib.unified_diff(a, b, 'run1', 'run2', lineterm='', n=1):
    print(d)
    n += 1
    if n > 120:
        print('... truncated')
        break
sys.exit(1)
