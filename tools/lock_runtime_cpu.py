"""Resolve the base CPU runtime; bind Torch hashes to its one explicit archive."""
import argparse
from pathlib import Path
import re
import subprocess
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def bind_torch_archive(text, url):
    digest = parse_qs(urlsplit(url).fragment).get('sha256', [''])[0]
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('the official Torch input needs its published SHA256')
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('torch @ ')]
    if len(starts) != 1 or lines[starts[0]].rstrip(' \\') != 'torch @ ' + url:
        raise ValueError('resolved Torch artifact differs from the platform input')
    start, end = starts[0], starts[0] + 1
    while end < len(lines) and lines[end].lstrip().startswith('--hash='):
        end += 1
    if not any('--hash=sha256:' + digest in line for line in lines[start + 1:end]):
        raise ValueError('resolved hashes omit the published Torch artifact hash')
    # Some uv versions emit all release hashes even for an explicit wheel URL.
    # Narrow that allowlist to the exact archive; never disable hash checking.
    lines[start:end] = ['torch @ ' + url + ' \\', '    --hash=sha256:' + digest]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uv', default='uv')
    parser.add_argument('--profile', choices=('base', 'hndl'), default='base')
    args = parser.parse_args()
    hndl = args.profile == 'hndl'
    output = ROOT / ('requirements-hndl-cpu.lock' if hndl else 'requirements-runtime-cpu.lock')
    project = 'profiles/hypergan-hndl/pyproject.toml' if hndl else 'pyproject.toml'
    omitted = ['--no-emit-package', 'particle-sliders-core', '--no-emit-package', 'hypergan']
    command = 'python tools/lock_runtime_cpu.py' + (' --profile hndl' if hndl else '')
    subprocess.run([args.uv, 'pip', 'compile', project, 'requirements-runtime-cpu.in',
        '--python-version', '3.12', '--python-platform', 'x86_64-unknown-linux-gnu',
        '--generate-hashes', *omitted, '--emit-index-url', '--default-index', 'https://pypi.org/simple',
        '--custom-compile-command', command, '--output-file', output.name], cwd=ROOT, check=True)
    requirement = next(line for line in (ROOT / 'requirements-runtime-cpu.in').read_text().splitlines()
                       if line.startswith('torch @ '))
    output.write_text(bind_torch_archive(output.read_text(), requirement.removeprefix('torch @ ')))


if __name__ == '__main__':
    main()
