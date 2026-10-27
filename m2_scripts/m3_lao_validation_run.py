"""Execute the predeclared, fresh-seed LAO versus saved-PNG-budget PGD experiment.

Creates an isolated directory, freezes source code and protocol before generation,
then runs data generation, two LAO controls, budget export, PGD, and PNG evaluation.
No checkpoint is trained, and no held-out score is used to choose a configuration.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


SD15_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
SD21_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
SOURCES = ["m2_stealth.py", "m3_lao_support.py", "m3_lao_run.py", "m3_lao_report.py",
           "m2_evaluate.py", "ControlVAE.py", "diff_latent_attack.py", "m2_device.py",
           "other_attacks.py", "network.py", "m2_data.py", "m3_lao_newdata.py",
           "m3_lao_budget.py", "m3_lao_pgd.py", "m3_lao_validate_report.py",
           "m3_lao_validation_run.py"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def check_sources(here, root, plan):
    for name, expected in plan["source_sha256"].items():
        if sha(here / name) != expected or sha(root / "frozen_code" / name) != expected:
            raise RuntimeError("Experiment source changed: " + name)
    if sha(root / "protocol.md") != plan["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")


def check_inputs(root):
    manifest_path = root / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "verified" or (manifest["n_generated"], manifest["n_real"]) != (50, 50):
        raise RuntimeError("The new data have not completed verification")
    if sha(root / "data/generation_plan.json") != manifest["plan_sha256"]:
        raise RuntimeError("The generation plan changed")
    if sha(root / "data/data_manifest.json") != sha(manifest_path):
        raise RuntimeError("Data manifests disagree")
    for row in manifest["generated"] + manifest["real"]:
        path = root / "data" / row["relative_path"]
        if not path.resolve().is_relative_to((root / "data").resolve()) or sha(path) != row["sha256"]:
            raise RuntimeError("Canonical input changed: " + row["relative_path"])


def require_fresh_stage(root, name):
    targets = {
        "data": ["data", "data_manifest.json"],
        "RAW_RECON": ["RAW_RECON"], "RAW_LAO": ["RAW_LAO"],
        "export": ["LAO8", "RECON8", "export_manifest.json"],
        "PGD": ["PGD10", "PGD40", "pgd_manifest.json"],
        "evaluation": ["eval"], "report": ["report"],
    }
    for relative in targets[name]:
        if (root / relative).exists():
            raise FileExistsError("Incomplete stage has retained output; refusing overwrite: " + str(root / relative))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--root", type=Path, help="new empty output directory, or an unchanged run to resume")
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip only completed stages; incomplete stages must have no outputs")
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parent
    reference = here / "results/m3_lao_dev/pilot_20261018_152125_odws2fqx/plan.json"
    prior = json.loads(reference.read_text())
    for name, expected in prior["source_sha256"].items():
        if sha(here / name) != expected:
            raise RuntimeError("The selected LAO implementation has changed: " + name)
    if args.resume and args.root is None:
        parser.error("--resume requires --root")
    parent = here / "results/m3_lao_validation"
    parent.mkdir(exist_ok=True, parents=True)
    if args.root is None:
        root = Path(tempfile.mkdtemp(prefix="validation_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_", dir=parent))
    else:
        root = args.root.resolve()
        if not args.resume:
            if root.exists() and any(root.iterdir()):
                parser.error("output directory is not empty")
            root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        plan = json.loads((root / "plan.json").read_text())
        if plan["device"] != args.device:
            parser.error("device differs from frozen plan")
        state = json.loads((root / "session.json").read_text())
        if state["status"] == "complete":
            print("Already complete: " + str(root), flush=True)
            return
        check_sources(here, root, plan)
    else:
        cache = Path.home() / ".cache/huggingface/hub"
        generator = cache / ("models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/" + SD15_REVISION)
        attack_model = cache / ("models--Manojb--stable-diffusion-2-1-base/snapshots/" + SD21_REVISION)
        for model in (generator, attack_model):
            for relative in ("unet/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.bin",
                             "text_encoder/pytorch_model.bin", "model_index.json"):
                if not (model / relative).is_file():
                    raise FileNotFoundError(model / relative)
        python = [sys.executable, "-B", "-u"]
        common = ["--device", args.device, "--images_root", str(root / "data/gen/test"),
                  "--max_images", "50", "--pretrained_diffusion_path", str(attack_model),
                  "--encoder_weights", "./checkpoints/Controlvae.pt", "--seed", "42",
                  "--encode_source", "clean", "--control_source", "clean", "--decode_path", "final",
                  "--surrogates", "E", "--evaluate", "E", "--loss", "margin", "--kappa", "3",
                  "--lpips_weight", "1", "--di_prob", "0", "--lao_eot", "none",
                  "--lao_weight_decay", "0", "--lao_diagnostics", "--lr", "0.01",
                  "--control_scale", "0.5", "--lao_residual_anchor"]
        names = ["RAW_RECON", "RAW_LAO", "RECON8", "LAO8", "PGD10", "PGD40"]
        commands = {
            "data": python + ["m3_lao_newdata.py", "--out", str(root / "data"), "--model_path", str(generator),
                              "--device", args.device, "--n", "50"],
            "RAW_RECON": python + ["m2_stealth.py", *common, "--iterations", "0", "--lao_quantization", "none",
                                   "--save_dir", str(root / "RAW_RECON")],
            "RAW_LAO": python + ["m2_stealth.py", *common, "--iterations", "10", "--lao_quantization", "ste",
                                 "--save_dir", str(root / "RAW_LAO")],
            "export": python + ["m3_lao_budget.py", "--source_run", str(root / "RAW_LAO"),
                                "--baseline_run", str(root / "RAW_RECON"), "--images_root", str(root / "data/gen/test"),
                                "--out_root", str(root), "--budget_pixels", "8"],
            "PGD": python + ["m3_lao_pgd.py", "--images_root", str(root / "data/gen/test"),
                             "--out_root", str(root), "--device", args.device, "--budget_pixels", "8", "--seed", "20261018"],
            "evaluation": python + ["m2_evaluate.py", "--device", args.device, "--real_dir", str(root / "data/real/test"),
                                    "--out", str(root / "eval"), "--detectors", "E,R,D,S",
                                    "--conditions", "none,jpeg75,resize05", "--runs", *[n + "=" + str(root / n) for n in names]],
            "report": python + ["m3_lao_validate_report.py", "--root", str(root), "--out", str(root / "report")],
        }
        protocol = here.parent / "docs/M3_LAO_HELDOUT_PROTOCOL_ZH.md"
        plan = {
            "created_at": datetime.now().astimezone().isoformat(), "device": args.device, "output": str(root),
            "n_generated": 50, "n_real": 50, "budget_pixels": 8,
            "budget_definition": "max absolute int16 difference between saved RGB uint8 PNG and canonical original <= 8; zero tolerance",
            "architecture_reference": str(reference), "architecture_reference_sha256": sha(reference),
            "model_training": False, "test_time_tuning": False,
            "quality_screen": {"psnr_min": 30.0, "lpips_max": 0.15},
            "primary": {"run": "LAO8", "comparator": "PGD40", "detector": "E", "condition": "none",
                        "metric": "quality-qualified conditional ASR paired difference", "bootstrap_seed": 20261018,
                        "bootstrap_resamples": 10000, "noninferiority_margin": 0.05,
                        "noninferiority_interval": "Bonferroni paired difference from 97.5% Clopper-Pearson discordant-category intervals"},
            "methods": {
                "RAW_RECON": "independent zero-step frozen residual-anchor reconstruction; unconstrained diagnostic",
                "RAW_LAO": "unchanged residual-anchor LAO, ten latent updates; unconstrained diagnostic",
                "RECON8": "integer-budget projection of RAW_RECON PNG",
                "LAO8": "integer-budget projection of RAW_LAO PNG; optimization is unaware of export projection",
                "PGD10": "ten clean-start pixel steps, alpha2/255, margin3+LPIPS1 on saved forward, final iterate",
                "PGD40": "40 steps per start, three starts, alpha2/255, maximize CE for AI label0, choose best saved E margin only"},
            "generation": {"model": "stable-diffusion-v1-5/stable-diffusion-v1-5", "revision": SD15_REVISION,
                           "seeds": list(range(5231000, 5231050)), "steps": 30, "guidance": 7.5,
                           "new_semantics": False, "generator_path": str(generator)},
            "attack_model": {"path": str(attack_model), "revision": SD21_REVISION},
            "checkpoint_sha256": prior["checkpoint_sha256"],
            "source_sha256": {name: sha(here / name) for name in SOURCES},
            "protocol_sha256": sha(protocol), "commands": commands,
        }
        for name, expected in plan["checkpoint_sha256"].items():
            if sha(here / "checkpoints" / name) != expected:
                raise RuntimeError("Checkpoint changed from development experiment: " + name)
        frozen = root / "frozen_code"
        frozen.mkdir()
        for name in SOURCES:
            shutil.copy2(here / name, frozen / name)
        shutil.copy2(protocol, root / "protocol.md")
        save(root / "plan.json", plan)
        state = {"status": "planned", "completed": [], "created_at": plan["created_at"]}
        save(root / "session.json", state)
        check_sources(here, root, plan)
    print("Frozen validation plan: " + str(root), flush=True)
    if args.plan_only:
        return
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    state.update(status="running", resumed_at=datetime.now().astimezone().isoformat())
    save(root / "session.json", state)
    try:
        for name, cmd in plan["commands"].items():
            if name in state["completed"]:
                continue
            check_sources(here, root, plan)
            if name != "data":
                check_inputs(root)
            require_fresh_stage(root, name)
            state.update(current=name)
            save(root / "session.json", state)
            print("Starting " + name + " at " + datetime.now().astimezone().isoformat(), flush=True)
            # Appending on a resume preserves the failed attempt's diagnostics.
            with (root / (name + ".log")).open("a") as log:
                proc = subprocess.Popen(cmd, cwd=here, stdout=log, stderr=subprocess.STDOUT, env=env)
                last_counts, last_print = None, time.monotonic()
                try:
                    while proc.poll() is None:
                        time.sleep(2)
                        if name == "data":
                            counts = {"generated": len(list((root / "data/gen512").glob("*.png")))}
                        else:
                            dirs = ["PGD10", "PGD40"] if name == "PGD" else [name]
                            counts = {d: len((root / d / "records.jsonl").read_text().splitlines())
                                      for d in dirs if (root / d / "records.jsonl").exists()}
                        if counts != last_counts or time.monotonic() - last_print >= 30:
                            print("  " + name + ": " + json.dumps(counts) + " / 50 images", flush=True)
                            last_counts, last_print = counts, time.monotonic()
                    if proc.returncode:
                        raise RuntimeError(f"{name} failed ({proc.returncode}); inspect {root / (name + '.log')}")
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
            save(root / "session.json", state)
        state.update(status="complete", finished_at=datetime.now().astimezone().isoformat())
        save(root / "session.json", state)
        print("Complete: " + str(root / "report"), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), finished_at=datetime.now().astimezone().isoformat())
        save(root / "session.json", state)
        raise


if __name__ == "__main__":
    main()
