"""Run a small, isolated LAO-only development experiment; never resumes old runs.

Run inside StealthDiffusion with the fit5230m2 environment. All variants use clean
encoding and a fixed clean control image. Model weights remain frozen. The stored
plan precedes execution; reports distinguish reconstruction from LAO's contribution.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


VARIANTS = {
    "R0": dict(iterations=0, decode_path="final", lao_quantization="none", control_scale=1.0),
    "R1": dict(decode_path="plain", lao_quantization="none", control_scale=1.0),
    "R2": dict(decode_path="final", lao_quantization="none", control_scale=1.0),
    "R3": dict(decode_path="final", lao_quantization="ste", control_scale=1.0),
    "G0": dict(iterations=0, decode_path="final", lao_quantization="none", control_scale=0.5),
    "G1": dict(decode_path="final", lao_quantization="ste", control_scale=0.5),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    p.add_argument("--max_images", type=int, choices=range(1, 11), default=10)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["R0", "R1", "R2", "R3"])
    p.add_argument("--model", default="Manojb/stable-diffusion-2-1-base")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--plan_only", action="store_true")
    p.add_argument("--residual_anchor", action="store_true",
                   help="anchor G0/G1 to the full-skip clean reconstruction; experimental residual rendering")
    args = p.parse_args(argv)
    if args.iterations <= 0 or args.iterations > 100:
        p.error("iterations must be 1..100; zero-step controls are selected by variant")
    if len(args.variants) != len(set(args.variants)):
        p.error("duplicate variants")
    if any(v in args.variants for v in ("R1", "R2", "R3")) and "R0" not in args.variants:
        p.error("R1/R2/R3 require their R0 reconstruction control")
    if "G1" in args.variants and "G0" not in args.variants:
        p.error("G1 requires its G0 reconstruction control")
    if args.residual_anchor and any(v not in ("G0", "G1") for v in args.variants):
        p.error("residual_anchor is defined only for the G0/G1 matched pair")
    here = Path(__file__).resolve().parent
    snapshot = args.snapshot.resolve()
    archive = json.loads((snapshot / "snapshot.json").read_text())
    if not archive.get("verified"):
        p.error("archive must have completed verification before new experiments")
    dev = here / "data15/gen/dev"
    all_inputs = sorted([x for x in dev.iterdir() if x.suffix.lower() in (".png", ".jpg", ".jpeg")],
                        key=lambda x: [int(v) if v.isdigit() else v for v in re.split(r"(\d+)", x.name)])
    inputs = all_inputs[:args.max_images]
    if len(inputs) != args.max_images:
        p.error("requested development images are unavailable")
    if not (here / "data15/real/dev").is_dir():
        p.error("real development split unavailable")
    if Path(args.model).is_dir():
        model_path = Path(args.model).resolve()
    else:
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(args.model, local_files_only=True)).resolve()
    real_inputs = sorted((here / "data15/real/dev").glob("*.png"))
    weight_names = ["Controlvae.pt", "efficientnet-b0.pth", "resnet50.pth", "deit.pth", "swin-t.pth"]
    model_files = [model_path / n for n in ["model_index.json", "scheduler/scheduler_config.json",
                   "unet/config.json", "unet/diffusion_pytorch_model.bin", "vae/config.json",
                   "vae/diffusion_pytorch_model.bin", "text_encoder/config.json", "text_encoder/pytorch_model.bin"]]
    for path in [*(here / "checkpoints" / n for n in weight_names), *model_files]:
        if not path.is_file():
            p.error("required local checkpoint missing: " + str(path))
    root = here / "results/m3_lao_dev"
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="pilot_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_", dir=root))
    common = ["--device", args.device, "--images_root", "data15/gen/dev", "--max_images", str(args.max_images),
              "--pretrained_diffusion_path", str(model_path), "--encoder_weights", "./checkpoints/Controlvae.pt",
              "--seed", "42", "--encode_source", "clean", "--control_source", "clean",
              "--surrogates", "E", "--evaluate", "E", "--loss", "margin", "--kappa", "3.0",
              "--lpips_weight", "1.0", "--di_prob", "0", "--lao_eot", "none",
              "--lao_weight_decay", "0", "--lao_diagnostics", "--lr", "0.01"]
    if args.residual_anchor:
        common.append("--lao_residual_anchor")
    commands = []
    for name in args.variants:
        cfg = {"iterations": args.iterations, **VARIANTS[name]}
        extra = [s for k, v in cfg.items() for s in ("--" + k, str(v))]
        commands.append((name, [sys.executable, "-B", "-u", "m2_stealth.py", *common, *extra,
                               "--save_dir", str(out / name)]))
    commands.append(("evaluation", [sys.executable, "-B", "-u", "m2_evaluate.py", "--device", args.device,
                     "--real_dir", "data15/real/dev", "--out", str(out / "eval"), "--detectors", "E,R,D,S",
                     "--conditions", "none", "--runs", *[n + "=" + str(out / n) for n in args.variants]]))
    commands.append(("report", [sys.executable, "-B", "-u", "m3_lao_report.py", "--eval_dir", str(out / "eval"),
                                "--run_root", str(out), "--out", str(out / "report"),
                                "--runs", ",".join(args.variants)]))
    sources = ["m2_stealth.py", "m3_lao_support.py", "m3_lao_run.py", "m3_lao_report.py", "m2_evaluate.py",
               "ControlVAE.py", "diff_latent_attack.py", "m2_device.py", "other_attacks.py"]
    plan = {"created_at": datetime.now().astimezone().isoformat(), "purpose": "Exploratory LAO-only development experiment",
            "snapshot": str(snapshot), "split": "development; not an untouched confirmatory test",
            "pixel_pgd": False, "model_training": False, "surrogate": "E", "held_out_from_loss": ["R", "D", "S"],
            "residual_anchor": args.residual_anchor,
            "quality_screen": {"psnr_min": 30.0, "lpips_max": 0.15, "meaning": "exploratory screen, not human semantic approval"},
            "input_sha256": {x.name: sha256(x) for x in inputs},
            "real_dev_sha256": {x.name: sha256(x) for x in real_inputs},
            "checkpoint_sha256": {n: sha256(here / "checkpoints" / n) for n in weight_names},
            "diffusion_model": {"requested": args.model, "resolved_local_path": str(model_path),
                                "file_sha256": {str(x.relative_to(model_path)): sha256(x) for x in model_files}},
            "source_sha256": {s: sha256(here / s) for s in sources},
            "commands": {name: cmd for name, cmd in commands}, "output": str(out)}
    save_json(out / "plan.json", plan)
    frozen = out / "frozen_code"
    frozen.mkdir()
    for source, expected in plan["source_sha256"].items():
        shutil.copy2(here / source, frozen / source)
        if sha256(frozen / source) != expected:
            raise RuntimeError("source changed while freezing experiment: " + source)
    save_json(frozen / "manifest.json", {"archived_at": datetime.now().astimezone().isoformat(),
              "verified_against_original_run_plan": True, "source_sha256": plan["source_sha256"]})
    print("Plan and output: " + str(out), flush=True)
    if args.plan_only:
        print("Plan only; no models loaded or optimizations executed.", flush=True)
        return
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    state = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "completed": []}
    save_json(out / "session.json", state)
    try:
        for name, cmd in commands:
            for source, expected in plan["source_sha256"].items():
                if sha256(here / source) != expected:
                    raise RuntimeError("source changed after experiment plan: " + source)
            state["current"] = name
            save_json(out / "session.json", state)
            print("Starting " + name + " at " + datetime.now().astimezone().isoformat(), flush=True)
            with (out / (name + ".log")).open("w") as log:
                proc = subprocess.Popen(cmd, cwd=here, stdout=log, stderr=subprocess.STDOUT, env=env)
                last_count = -1
                try:
                    while proc.poll() is None:
                        time.sleep(2)
                        rec = out / name / "records.jsonl"
                        if rec.exists():
                            count = len(rec.read_text().splitlines())
                            if count != last_count:
                                print(f"  {name}: {count}/{args.max_images} images complete", flush=True)
                                last_count = count
                    if proc.returncode:
                        raise RuntimeError(f"{name} failed ({proc.returncode}); see {out / (name + '.log')}")
                except BaseException:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    raise
            state["completed"].append(name)
            save_json(out / "session.json", state)
        state.update(status="complete", finished_at=datetime.now().astimezone().isoformat())
        save_json(out / "session.json", state)
        print("Complete: " + str(out / "report"), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), finished_at=datetime.now().astimezone().isoformat())
        save_json(out / "session.json", state)
        raise


if __name__ == "__main__":
    main()
