"""Machine-local bridge to the unchanged verified MOSS training command."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/ml2/particlesliders-working/moss_training/train_concept.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def verify(output, steps):
    result = json.loads((output / 'result.json').read_text())
    launcher = json.loads((output / 'launcher.json').read_text())
    if result.get('status') != 'completed' or result.get('completed_steps') != steps:
        raise ValueError('MOSS did not complete the requested native updates')
    if launcher.get('exit_code') != 0 or not launcher.get('worker_reaped'):
        raise ValueError('MOSS worker did not finish cleanly')
    checkpoint = output / f'checkpoints/step-{steps:06d}'
    receipt = json.loads((checkpoint / 'checkpoint.json').read_text())
    if receipt['step'] != steps or sha(checkpoint / 'training.pt') != receipt['sha256']:
        raise ValueError('MOSS checkpoint bytes or completed step changed')
    adapter = output / 'adapter'
    manifest = json.loads((adapter / 'slider.json').read_text())
    if manifest['completed_steps'] != steps or sha(adapter / 'adapter.safetensors') != manifest['adapter_sha256']:
        raise ValueError('MOSS adapter bytes or completed step changed')
    return dict(result, checkpoint=str(checkpoint), adapter=str(adapter), lr_search=False,
                objective='pg09_gan_only', native_source=str(SOURCE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-every', type=int, default=32)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    profile = json.loads((ROOT / 'training_profile.local-pg094.json').read_text())
    if sha(SOURCE) != profile['source_sha256']:
        raise ValueError('local MOSS source changed; reverify before updating profile')
    config = json.loads(args.config.read_text())
    native = args.output.resolve() / 'native'
    command = [profile['runtime']['python'], '-I', str(SOURCE), '--gpu', '0', '--config', str(args.config.resolve()),
               '--output', str(native), '--checkpoint-every', str(args.checkpoint_every)]
    if args.resume:
        command += ['--resume', str(args.resume.resolve())]
    if args.check:
        return subprocess.run([*command, '--check'], check=True).returncode
    args.output.mkdir(parents=True, exist_ok=False)
    child = None
    try:
        parent = os.getpid()
        def guard():
            if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0):
                raise OSError('cannot set native launcher parent-death guard')
            if os.getppid() != parent:
                os._exit(125)
        child = subprocess.Popen(command, preexec_fn=guard)
        while child.poll() is None:
            if (native / 'metrics.jsonl').exists():
                (args.output / 'metrics.jsonl').write_bytes((native / 'metrics.jsonl').read_bytes())
            time.sleep(1)
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
        result = verify(native, config['steps'])
        (args.output / 'metrics.jsonl').write_bytes((native / 'metrics.jsonl').read_bytes())
        write(args.output / 'result.json', result)
    except BaseException as error:
        write(args.output / 'result.json', dict(status='failed', error=repr(error)))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
