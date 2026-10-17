"""Two GPU dispatch with resource guards; stops only this batch's process groups."""

import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path("/mnt/nvme/sliders/runs/moss-nonlighting-campaign-v1")
PROJECT = Path("/mnt/nvme/sliders/moss-image-sliders-campaign")
RUNNER = str(PROJECT / "tools/train_campaign.py")
PYTHON = "/mnt/nvme/sliders/moss-image-sliders/.venv/bin/python"
start = time.time()
active = {}
completed = {}
halted = {}
locks = []
samples = []
events = []
seen_kernel = set()
kernel_access = True


def event(kind, **kwargs):
    item = {
        "event": kind,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **kwargs,
    }
    events.append(item)
    with (ROOT / "monitor-events.jsonl").open("a") as f:
        f.write(json.dumps(item) + "\n")
    print(json.dumps(item), flush=True)


def gpu_status():
    query = "index,uuid,temperature.gpu,power.draw,power.limit,memory.used,memory.free,utilization.gpu,ecc.errors.uncorrected.volatile.total"
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    result = {}
    for line in lines.splitlines():
        p = [v.strip() for v in line.split(",")]
        result[int(p[0])] = {
            "index": int(p[0]),
            "uuid": p[1],
            "temperature_c": float(p[2]),
            "power_w": float(p[3]),
            "power_limit_w": float(p[4]),
            "used_mib": float(p[5]),
            "free_mib": float(p[6]),
            "utilization_percent": float(p[7]),
            "uncorrected_ecc": p[8],
        }
    return result


def processes():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    result = []
    for line in output.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3 and p[1].isdigit():
            result.append((p[0], int(p[1]), float(p[2]) if p[2].isdigit() else 0))
    return result


def stop(gpu, reason):
    halted[gpu] = reason
    if gpu not in active:
        return
    process, job, log, launched = active[gpu]
    event("job_stopped", gpu=gpu, pid=process.pid, concept=job["concept"], reason=reason)
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


if (
    not os.path.ismount("/mnt/nvme")
    or subprocess.check_output(
        ["findmnt", "-no", "SOURCE", "--target", "/mnt/nvme"], text=True
    ).strip()
    != "/dev/nvme0n1p2"
):
    raise RuntimeError("Expected NVMe is not mounted")
lock_root = Path("/mnt/nvme/sliders/locks")
lock_root.mkdir(exist_ok=True)
for gpu in [0, 1]:
    f = (lock_root / f"gpu-{gpu}.lock").open("a")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(f)
baseline = gpu_status()
existing = processes()
(ROOT / "gpu-baseline.json").write_text(
    json.dumps({"gpus": baseline, "existing_processes": existing}, indent=2) + "\n"
)
event("monitor_launched", pid=os.getpid(), gpus=[0, 1])
last_kernel = 0
try:
    while True:
        manifest = json.loads((ROOT / "batch.json").read_text())
        guards = manifest["guards"]
        try:
            state = gpu_status()
            current = processes()
        except Exception as error:
            for gpu in list(active):
                stop(gpu, "GPU monitoring failed: " + str(error))
            raise
        now = time.time()
        record = {"elapsed_seconds": now - start, "gpus": state}
        samples.append(record)
        with (ROOT / "gpu-telemetry.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        mem = {
            line.split(":")[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        }
        host_available_mib = mem["MemAvailable"] / 1024
        own = {v[0].pid for v in active.values()}
        for gpu, info in state.items():
            if gpu not in active:
                continue
            reason = None
            if info["temperature_c"] >= guards["max_gpu_temperature_c"]:
                reason = "Temperature threshold reached"
            if info["free_mib"] < guards["min_free_vram_mib"]:
                reason = "Free VRAM below reserve"
            if info["power_w"] > info["power_limit_w"] * guards["max_gpu_power_ratio"]:
                reason = "Power draw exceeded guard"
            if info["uncorrected_ecc"].isdigit() and int(info["uncorrected_ecc"]) > int(
                baseline[gpu]["uncorrected_ecc"]
                if baseline[gpu]["uncorrected_ecc"].isdigit()
                else 0
            ):
                reason = "New uncorrected ECC error"
            if host_available_mib < guards["min_host_available_memory_mib"]:
                reason = "Host memory reserve exhausted"
            if any(
                uuid == info["uuid"]
                and pid not in own
                and (uuid, pid, used) not in existing
                and used > 2048
                and not any(old[0] == uuid and old[1] == pid for old in existing)
                for uuid, pid, used in current
            ):
                reason = "New competing GPU workload detected"
            if reason:
                stop(gpu, reason)
        if now - last_kernel >= 10:
            last_kernel = now
            result = subprocess.run(
                [
                    "journalctl",
                    "-k",
                    "--since",
                    "@" + str(int(start)),
                    "--no-pager",
                    "-g",
                    "NVRM: Xid|GPU has fallen|nvidia.*[Ee]rror",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode not in [0, 1] or "permission" in result.stderr.lower():
                kernel_access = False
            for line in result.stdout.splitlines():
                if (
                    re.search(r"Xid|GPU has fallen|nvidia.*[Ee]rror", line)
                    and line not in seen_kernel
                ):
                    seen_kernel.add(line)
                    with (ROOT / "kernel-gpu-errors.log").open("a") as f:
                        f.write(line + "\n")
                    affected = [
                        g
                        for g in active
                        if state[g]["uuid"] in line or state[g]["uuid"][:12] in line
                    ]
                    for gpu in affected or list(active):
                        stop(gpu, "New kernel GPU error")
        for gpu, (process, job, log, launched) in list(active.items()):
            if (
                process.poll() is None
                and now - launched > manifest["budget"]["per_job_timeout_seconds"]
            ):
                stop(gpu, "Per-job timeout")
            code = process.poll()
            if code is not None:
                log.close()
                completed[job["concept"]] = {
                    "gpu": gpu,
                    "pid": process.pid,
                    "exit_code": code,
                    "wall_seconds": now - launched,
                }
                event("job_exited", concept=job["concept"], **completed[job["concept"]])
                del active[gpu]
                if code != 0:
                    halted[gpu] = halted.get(gpu, "Job error; remaining GPU queue stopped")
        if now - start > manifest["budget"]["batch_timeout_seconds"]:
            for gpu in list(active):
                stop(gpu, "Batch timeout")
            break
        for gpu in [0, 1]:
            if gpu in active or gpu in halted:
                continue
            jobs = [
                job
                for job in manifest["jobs"]
                if job["gpu"] == gpu and job["concept"] not in completed
            ]
            if not jobs:
                continue
            if (Path(jobs[0]["output"]) / "progress.json").exists():
                old = json.loads((Path(jobs[0]["output"]) / "progress.json").read_text())
                if old.get("status") == "completed":
                    completed[jobs[0]["concept"]] = {
                        "gpu": gpu,
                        "pid": old.get("pid"),
                        "exit_code": 0,
                        "recovered_completed_receipt": True,
                    }
                    continue
            if state[gpu]["free_mib"] < 24 * 1024:
                halted[gpu] = "Less than 24 GiB free VRAM before dispatch"
                event("dispatch_blocked", gpu=gpu, reason=halted[gpu])
                continue
            job = jobs[0]
            env = os.environ.copy()
            env.update(
                PYTHONPATH=str(PROJECT / "src"),
                OMP_NUM_THREADS="4",
                CUDA_VISIBLE_DEVICES=str(gpu),
                HF_HOME="/mnt/nvme/sliders/cache/huggingface",
                XDG_CACHE_HOME="/mnt/nvme/sliders/cache/xdg",
                TMPDIR="/mnt/nvme/sliders/tmp",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                GRADIO_ANALYTICS_ENABLED="False",
                CUDA_CACHE_PATH="/mnt/nvme/sliders/cache/cuda",
                TRITON_CACHE_DIR="/mnt/nvme/sliders/cache/triton",
                TORCH_HOME="/mnt/nvme/sliders/cache/torch",
                TORCHINDUCTOR_CACHE_DIR="/mnt/nvme/sliders/cache/torchinductor",
            )
            log = (Path(job["output"]) / "run.log").open("a")
            process = subprocess.Popen(
                [PYTHON, "-u", RUNNER, job["recipe"]],
                cwd=PROJECT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[gpu] = (process, job, log, time.time())
            event(
                "job_launched",
                concept=job["concept"],
                gpu=gpu,
                pid=process.pid,
                log=str(Path(job["output"]) / "run.log"),
            )
        if not active and len(completed) == len(manifest["jobs"]):
            break
        if not active and halted:
            break
        time.sleep(guards["poll_seconds"])
except BaseException as error:
    event("monitor_error", error=type(error).__name__ + ": " + str(error))
    for gpu in list(active):
        stop(gpu, "Monitor interrupted or failed")
    raise
finally:
    summary = {
        "status": "completed"
        if not halted and len(completed) == len(manifest["jobs"])
        else "stopped_or_incomplete",
        "wall_seconds": time.time() - start,
        "jobs": completed,
        "halted_gpus": halted,
        "kernel_error_log_access": kernel_access,
        "kernel_gpu_errors": len(seen_kernel),
        "telemetry_samples": len(samples),
        "gpu_summary": {},
    }
    for gpu in [0, 1]:
        rows = [x["gpus"][gpu] for x in samples if gpu in x["gpus"]]
        if rows:
            summary["gpu_summary"][gpu] = {
                "max_temperature_c": max(x["temperature_c"] for x in rows),
                "max_power_w": max(x["power_w"] for x in rows),
                "max_used_mib": max(x["used_mib"] for x in rows),
                "min_free_mib": min(x["free_mib"] for x in rows),
                "max_utilization_percent": max(x["utilization_percent"] for x in rows),
                "mean_utilization_percent": sum(x["utilization_percent"] for x in rows) / len(rows),
            }
    (ROOT / "monitor-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    event("monitor_finished", **summary)
