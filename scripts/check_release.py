#!/usr/bin/env python3
"""Static candidate checks; not a security audit, license grant or GPU test."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

SKIP_DIRS={'.git','__pycache__','.pytest_cache','outputs','debug_files','.venv','checkpoints'}
PRIVATE_SUFFIXES={'.pt','.pth','.ckpt','.safetensors','.pkl','.pickle','.csv','.xlsx','.png','.jpg','.jpeg','.webp'}
SECRET_PATTERNS=[
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{50,}\b'),
    re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
]

def candidate_files(root):
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            yield path
        elif path.is_file():
            yield path

def inspect(root):
    findings=[]
    count=0
    for path in candidate_files(root):
        count+=1
        rel=str(path.relative_to(root))
        if path.is_symlink():
            findings.append([rel,'symlink is not allowed in the release candidate'])
            continue
        if path.suffix.lower() in PRIVATE_SUFFIXES or path.name.startswith('.env'):
            findings.append([rel,'data/weight/environment file requires explicit review'])
        if path.stat().st_size>10*1024*1024:
            findings.append([rel,'file exceeds source-package size budget'])
            continue
        try:
            value=path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            findings.append([rel,'binary file requires explicit review'])
            continue
        if path.suffix=='.py':
            try: ast.parse(value,filename=rel)
            except SyntaxError as exc: findings.append([rel,f'Python syntax error at line {exc.lineno}'])
        if path.suffix=='.json':
            try: json.loads(value)
            except json.JSONDecodeError as exc: findings.append([rel,f'JSON syntax error at line {exc.lineno}'])
        for index,line in enumerate(value.splitlines(),1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                findings.append([rel,f'possible credential at line {index}; value not printed'])
            # Match actual home/mount paths, not generic /path/to examples.
            if re.search(r'/(?:home|data1|data2)/[A-Za-z0-9_.-]+/',line):
                findings.append([rel,f'machine-specific path at line {index}'])
    return {'checked_files':count,'findings':findings,
            'scope':'Static text/source scan only; no guarantee of absence of secrets or license issues.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args()
    result=inspect(args.root)
    print(json.dumps(result,indent=2))
    return bool(result['findings'])

if __name__=='__main__':
    sys.exit(main())
