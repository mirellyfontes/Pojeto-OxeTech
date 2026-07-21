import subprocess, sys
from pathlib import Path

req = Path(r'c:/Users/Mirelly/Documents/Pojeto OxeTech/requirements.txt')
cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(req)]
print('Running:', ' '.join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
print(result.stderr, file=sys.stderr)
raise SystemExit(result.returncode)
