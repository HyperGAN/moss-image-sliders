"""Private, resumable MOSS campaign. Run only under the resource-guarding supervisor."""

import argparse
from collections import OrderedDict
import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
import torch
from diffusers import AutoencoderKLFlux2

from moss_sliders import MODEL_ID, MODEL_REVISION
from moss_sliders.backend import load_pipeline
from moss_sliders.campaign import (
    StateSampler,
    atomic_json,
    cached_backward,
    digest,
    edit_diagnostics,
    latest_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
    target_bundle,
)
from moss_sliders.lora import Slider
from moss_sliders.training import checkpoint_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe")
    args = parser.parse_args()
    recipe_path = Path(args.recipe).resolve()
    root = recipe_path.parent
    recipe = json.loads(recipe_path.read_text())
    identity = digest(recipe)
    settings = recipe["settings"]
    if recipe["concept"] not in ("bad-intent", "afterimage", "final-form"):
        raise ValueError("Only the three authorized non-lighting concepts are allowed")
    if (
        not os.path.ismount("/mnt/nvme")
        or subprocess.check_output(
            ["findmnt", "-no", "SOURCE", "--target", "/mnt/nvme"], text=True
        ).strip()
        != "/dev/nvme0n1p2"
    ):
        raise RuntimeError("Expected NVMe is not mounted")
    if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        raise RuntimeError("Less than 24 GiB free VRAM before launch")
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = (
        json.loads((root / "progress.json").read_text())
        if (root / "progress.json").exists()
        else {}
    )
    if previous.get("status") == "completed":
        raise RuntimeError("Already completed; refusing duplicate job")
    start = time.monotonic()
    stop_requested = False
    cooling_seconds = 0.0
    last_temperature_check = 0.0
    step = 0
    progress = dict(
        status="running",
        concept=recipe["concept"],
        pid=os.getpid(),
        physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        identity=identity,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_revision=MODEL_REVISION,
        recipe=str(recipe_path),
        latest_step=0,
        attempt=previous.get("attempt", 0) + 1,
    )
    history = root / f"attempt-{progress['attempt']:02d}.jsonl"

    def event(kind, **values):
        progress.update(values)
        progress.update(elapsed_seconds=time.monotonic() - start, cooling_seconds=cooling_seconds)
        atomic_json(root / "progress.json", progress)
        item = dict(
            progress, event=kind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        with history.open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def thermal_gate(*_):
        nonlocal last_temperature_check, cooling_seconds
        if time.monotonic() - last_temperature_check < 1:
            return
        last_temperature_check = time.monotonic()

        def temperature():
            return float(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
                        "--query-gpu=temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                ).strip()
            )

        current = temperature()
        if current >= 82:
            raise RuntimeError("82 C hard temperature cutoff")
        if current >= 76:
            torch.cuda.synchronize()
            before = time.monotonic()
            event("cooling_pause", cooling_active=True, temperature_c=current)
            while current > 70:
                if time.monotonic() - before > 180:
                    raise RuntimeError("Cooling pause exceeded 180 seconds")
                time.sleep(1)
                current = temperature()
            cooling_seconds += time.monotonic() - before
            event("cooling_resumed", cooling_active=False, temperature_c=current)

    def finite_hook(module, args, result):
        value = result.sample if hasattr(result, "sample") else result[0]
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite model output")

    def checksum(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    try:
        torch.set_num_threads(4)
        torch.manual_seed(settings["seed"])
        event("launched", stage="loading", torch=torch.__version__)
        pipe = load_pipeline(device="cuda:0", local_files_only=True, training=True)
        pipe.set_progress_bar_config(disable=True)
        pipe.text_encoder.register_forward_pre_hook(thermal_gate)
        rows = recipe["training_rows"]
        dev = recipe["development_rows"]
        prompts = list(
            dict.fromkeys([r[k] for r in rows + dev for k in ("neutral", "positive")] + [""])
        )
        encoded_path = root / "embeddings.safetensors"
        prompt_hash = digest(prompts)
        if encoded_path.exists():
            metadata = json.loads((root / "embeddings.json").read_text())
            if metadata["prompts_sha256"] != prompt_hash or metadata["sha256"] != checksum(
                encoded_path
            ):
                raise ValueError("Encoded prompt cache changed")
            tensors = load_file(str(encoded_path))
            encoded = {p: (tensors[f"e{i}"], tensors[f"m{i}"]) for i, p in enumerate(prompts)}
        else:
            encoded = {}
            with torch.no_grad():
                for i, prompt in enumerate(prompts):
                    text = pipe.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    tokens = pipe.tokenizer(text, return_tensors="pt", truncation=False).to(
                        "cuda:0"
                    )
                    if tokens.input_ids.shape[-1] > settings["max_sequence_length"]:
                        raise ValueError("Prompt exceeds verified token budget")
                    hidden = pipe.text_encoder(
                        **tokens, use_cache=False, return_dict=True
                    ).last_hidden_state
                    encoded[prompt] = (hidden.detach().cpu(), tokens.attention_mask.bool().cpu())
                    event(
                        "prompt_encoded",
                        stage="encoding",
                        encoded=i + 1,
                        total_prompts=len(prompts),
                    )
                    if stop_requested:
                        raise InterruptedError("Stopped at safe prompt boundary")
            tensors = {
                f"{kind}{i}": encoded[p][j].contiguous()
                for i, p in enumerate(prompts)
                for j, kind in enumerate(("e", "m"))
            }
            temporary = encoded_path.with_suffix(".pending")
            save_file(tensors, str(temporary))
            temporary.replace(encoded_path)
            atomic_json(
                root / "embeddings.json",
                dict(prompts_sha256=prompt_hash, sha256=checksum(encoded_path)),
            )
            del hidden, tokens
        del tensors
        pipe.text_encoder = None
        pipe.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        pipe.transformer.to("cuda:0").eval()
        encoded = {p: tuple(t.to("cuda:0") for t in pair) for p, pair in encoded.items()}
        slider = Slider(pipe.transformer, rank=settings["rank"], alpha=settings["alpha"])
        pipe.transformer.register_forward_pre_hook(thermal_gate)
        pipe.transformer.register_forward_hook(finite_hook)
        source_initial = {k: v.detach().cpu().clone() for k, v in slider.state_dict().items()}
        cache = root / "targets"
        cache.mkdir(exist_ok=True)
        bundles = [(row, seed, "train") for row in rows for seed in row["seeds"]]
        bundles += [(row, recipe["evaluation_seed"], "dev") for row in dev]
        records = []
        preparation_started = time.monotonic()
        prepared_new = 0
        for i, (row, seed, split) in enumerate(bundles):
            path = cache / f"{split}-{row['id']}-{seed}.safetensors"
            sidecar = path.with_suffix(".json")
            target_identity = digest(dict(identity=identity, row=row["id"], seed=seed, split=split))
            if path.exists() and sidecar.exists():
                meta = json.loads(sidecar.read_text())
                if meta["identity"] != target_identity or meta["sha256"] != checksum(path):
                    raise ValueError("Frozen target checksum/identity mismatch")
            else:
                before = time.monotonic()
                data = target_bundle(
                    pipe,
                    slider,
                    encoded[row["neutral"]],
                    encoded[row["positive"]],
                    seed,
                    resolution=settings["resolution"],
                    steps=settings["target_steps"],
                )
                temporary = path.with_suffix(".pending")
                save_file(data, str(temporary))
                temporary.replace(path)
                atomic_json(sidecar, dict(identity=target_identity, sha256=checksum(path)))
                prepared_new += 1
                event(
                    "target_bundle_saved",
                    stage="preparing_targets",
                    bundles_complete=i + 1,
                    bundles_total=len(bundles),
                    states_complete=(i + 1) * 2 * settings["target_steps"],
                    bundle_seconds=time.monotonic() - before,
                    prepared_new=prepared_new,
                    preparation_wall_seconds=time.monotonic() - preparation_started,
                )
            records.append(dict(path=str(path), row=row["id"], seed=seed, split=split))
            if stop_requested:
                raise InterruptedError("Stopped at safe frozen-target boundary")
        atomic_json(root / "target-index.json", dict(identity=identity, records=records))
        train_records = [r for r in records if r["split"] == "train"]
        dev_records = [r for r in records if r["split"] == "dev"]
        positions = settings["target_steps"]
        sampler = StateSampler(len(train_records) * 2 * positions, settings["seed"] + 1)
        optimizer = torch.optim.AdamW(slider.parameters(), lr=settings["lr"])
        checkpoint_blocks(pipe.transformer)
        checkpoints = root / "checkpoints"
        selected = latest_training_checkpoint(checkpoints, identity)
        if selected is not None:
            step = restore_training_checkpoint(selected, slider, optimizer, sampler, identity)
            event("checkpoint_restored", latest_step=step)
        else:
            slider.load_state_dict(source_initial)
            save_training_checkpoint(checkpoints, 0, slider, optimizer, sampler, identity)
        del source_initial
        row_by_id = {r["id"]: r for r in rows + dev}
        loaded = OrderedDict()

        def sample_from(record, path_index, position):
            path = record["path"]
            if path not in loaded:
                loaded[path] = load_file(path)
                while len(loaded) > 4:
                    loaded.popitem(last=False)
            data = loaded[path]
            sample = {
                key: data[key][path_index, position].to("cuda:0")
                for key in ("latent", "base", "delta")
            }
            sample["time"] = data["time"][position].to("cuda:0")
            return sample, encoded[row_by_id[record["row"]]["neutral"]]

        def fixtures():
            for record in dev_records:
                for path_index in (0, 1):
                    for position in (0, 3, 6, 9):
                        yield sample_from(record, path_index, position)

        def diagnostics(at_step):
            report = edit_diagnostics(pipe.transformer, slider, fixtures())
            directory = root / "diagnostics"
            directory.mkdir(exist_ok=True)
            atomic_json(directory / f"step-{at_step:06d}.json", report)
            best_path = root / "BEST-DIAGNOSTIC.json"
            old = json.loads(best_path.read_text()) if best_path.exists() else None
            score = report["1"]["relative_edit_mse"]
            if old is None or score < old["relative_edit_mse"]:
                atomic_json(
                    best_path,
                    dict(
                        step=at_step,
                        relative_edit_mse=score,
                        criterion="fixed development positive edit MSE; not a perceptual winner",
                    ),
                )
            event(
                "fixed_diagnostics",
                stage="evaluating",
                diagnostic_step=at_step,
                edit_diagnostics=report,
            )
            return report

        def render(at_step, baselines=False):
            loaded.clear()
            gc.collect()
            torch.cuda.empty_cache()
            vae = (
                AutoencoderKLFlux2.from_pretrained(
                    MODEL_ID,
                    subfolder="vae",
                    revision=MODEL_REVISION,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                )
                .to("cuda:0")
                .eval()
            )
            vae.requires_grad_(False)
            original_decode = vae.decode

            def checked_decode(*args, _decode=original_decode, _vae=vae, **kwargs):
                thermal_gate()
                result = _decode(*args, **kwargs)
                finite_hook(_vae, args, result)
                return result

            vae.decode = checked_decode
            pipe.register_modules(vae=vae)
            output = root / "samples" / ("baselines" if baselines else f"step-{at_step:06d}")
            output.mkdir(parents=True, exist_ok=True)
            for row_index, row in enumerate(dev):
                variants = (
                    [
                        ("off", row["neutral"], 0, 3),
                        ("direct-positive", row["positive"], 0, 3),
                        ("direct-positive-cfg1", row["positive"], 0, 1),
                    ]
                    if baselines
                    else [("minus", row["neutral"], -1, 3), ("plus", row["neutral"], 1, 3)]
                )
                for label, prompt, scale, cfg in variants:
                    destination = output / f"case-{row_index}-{label}.png"
                    if destination.exists():
                        continue
                    emb, mask = encoded[prompt]
                    neg, negmask = encoded[""]
                    before = time.monotonic()
                    with slider.at(scale), torch.inference_mode():
                        image = pipe(
                            prompt_embeds=emb,
                            prompt_attention_mask=mask,
                            negative_prompt_embeds=neg,
                            negative_prompt_attention_mask=negmask,
                            height=settings["resolution"],
                            width=settings["resolution"],
                            num_inference_steps=50,
                            guidance_scale=cfg,
                            uncond_skip_layers=[22] if cfg != 1 else [],
                            generator=torch.Generator("cpu").manual_seed(recipe["evaluation_seed"]),
                        ).images[0]
                    if not np.isfinite(np.asarray(image)).all():
                        raise RuntimeError("Non-finite rendered image")
                    image.save(destination)
                    atomic_json(
                        destination.with_suffix(".json"),
                        dict(
                            step=at_step,
                            scale=scale,
                            cfg=cfg,
                            prompt=prompt,
                            seed=recipe["evaluation_seed"],
                            sha256=checksum(destination),
                            inference_steps=50,
                            width=512,
                            height=512,
                            model_revision=MODEL_REVISION,
                        ),
                    )
                    event(
                        "image_saved",
                        stage="evaluating",
                        file=str(destination),
                        image_seconds=time.monotonic() - before,
                    )
                if not baselines:
                    labels = ["Off", "Frozen positive prompt", "Slider -1", "Slider +1"]
                    paths = [
                        root / "samples/baselines" / f"case-{row_index}-off.png",
                        root / "samples/baselines" / f"case-{row_index}-direct-positive.png",
                        output / f"case-{row_index}-minus.png",
                        output / f"case-{row_index}-plus.png",
                    ]
                    sheet = Image.new("RGB", (2048, 576), "#121822")
                    draw = ImageDraw.Draw(sheet)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
                    draw.text(
                        (12, 5),
                        f"{recipe['concept']} | update {at_step} | held-out seed {recipe['evaluation_seed']}",
                        font=font,
                        fill="white",
                    )
                    for column, (path, label) in enumerate(zip(paths, labels)):
                        draw.text((column * 512 + 12, 32), label, font=font, fill="white")
                        with Image.open(path) as image:
                            sheet.paste(image, (column * 512, 64))
                    sheet.save(output / f"case-{row_index}-comparison.png")
            pipe.register_modules(vae=None)
            # Break the decode closure's self-cycle before returning memory to the allocator.
            vae.decode = original_decode
            del original_decode, checked_decode, vae
            gc.collect()
            torch.cuda.empty_cache()

        if step == 0 and not (root / "diagnostics/step-000000.json").exists():
            diagnostics(0)
        render(0, baselines=True)
        metrics_path = root / "metrics.jsonl"
        if metrics_path.exists():
            # Only complete optimizer updates in the selected checkpoint are authoritative.
            committed = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            committed = [r for r in committed if r["step"] <= step]
            metrics_path.write_text("".join(json.dumps(r) + "\n" for r in committed))
        for existing_step in settings["evaluation_updates"]:
            if existing_step == step:
                diagnostics(step)
                render(step)
        event(
            "training_started",
            stage="training",
            latest_step=step,
            effective_batch=8,
            target_states=len(train_records) * 2 * positions,
        )
        stage_start = time.monotonic()
        stage_cooling = cooling_seconds
        start_step = step
        while step < settings["updates"] and not stop_requested:
            before = time.monotonic()
            cooling_before = cooling_seconds
            indices = sampler.take(settings["effective_batch"])
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            teacher_energy = 0.0
            exposure = []
            for index in indices:
                bundle_index, local_index = divmod(index, 2 * positions)
                path_index, position = divmod(local_index, positions)
                record = train_records[bundle_index]
                sample, context = sample_from(record, path_index, position)
                loss += (
                    cached_backward(
                        pipe.transformer, slider, sample, context, settings["effective_batch"]
                    )
                    / settings["effective_batch"]
                )
                teacher_energy += (
                    float(sample["delta"].square().mean()) / settings["effective_batch"]
                )
                exposure.append([record["row"], record["seed"], path_index, position])
            norm = torch.nn.utils.clip_grad_norm_(slider.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            values = dict(
                step=step,
                loss=loss,
                grad_norm=float(norm),
                teacher_delta_mse=teacher_energy,
                relative_signed_mse=loss / max(teacher_energy, 1e-30),
                exposure=exposure,
                update_wall_seconds=time.monotonic() - before,
                update_cooling_seconds=cooling_seconds - cooling_before,
            )
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(values) + "\n")
            event("optimizer_step", stage="training", latest_step=step, **values)
            if step % 10 == 0 or stop_requested:
                save_training_checkpoint(
                    checkpoints, step, slider, optimizer, sampler, identity, values
                )
                event("checkpoint_saved", checkpoint_step=step)
            if step == start_step + 20:
                wall = time.monotonic() - stage_start
                active = wall - (cooling_seconds - stage_cooling)
                receipt = dict(
                    updates=20,
                    effective_batch=8,
                    wall_seconds=wall,
                    active_seconds=active,
                    wall_seconds_per_update=wall / 20,
                    active_seconds_per_update=active / 20,
                    remaining_training_wall_hours=(settings["updates"] - step) * wall / 20 / 3600,
                    note="Excludes target preparation and checkpoint evaluations; thermal behavior can change",
                )
                atomic_json(root / f"measured-stage-from-{start_step:06d}.json", receipt)
                event("measured_stage_completed", **receipt)
                if active / 20 * settings["updates"] > 16.5 * 3600:
                    raise RuntimeError(
                        "Measured active training estimate exceeds 16.5 hours per concept; review budget"
                    )
            if step in settings["evaluation_updates"]:
                diagnostics(step)
                render(step)
                event("checkpoint_evaluation_completed", evaluated_step=step)
        if stop_requested:
            if not (checkpoints / f"step-{step:06d}").exists():
                save_training_checkpoint(checkpoints, step, slider, optimizer, sampler, identity)
            event("stopped_at_safe_boundary", status="stopped", latest_step=step)
        else:
            event(
                "completed",
                status="completed",
                stage="done",
                latest_step=step,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
    except BaseException as error:
        event(
            "failed",
            status="stopped" if isinstance(error, InterruptedError) else "failed",
            error=type(error).__name__ + ": " + str(error),
            latest_step=step,
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
