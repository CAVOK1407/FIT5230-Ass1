"""Archive-linked development selection, then one frozen second-batch validation.

Run development first. Its freeze.json is required by the validation phase.
No previous output directory is overwritten and no held-out score selects a method.
"""
import argparse
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image


SNAPSHOT = "before_lao_low_distortion_v2_20261018_165859"
FIRST_RUN = "validation_20261018_154651_tm5zwx4p"
DEV_REFERENCE = "pilot_20261018_152125_odws2fqx"
NAMES = ["OLD_RECON8", "OLD_LAO8", "NEW_RECON8", "NEW_LAO8", "PGD10", "PGD10_MATCHED"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def source_files(here):
    # Include helper code as well as entry points. Results and test data are not code.
    paths = list(here.glob("*.py"))
    for folder in (here / "lossers",):
        paths.extend(folder.rglob("*.py"))
    return sorted(set(paths))


def check_sources(plan):
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("Runtime code changed: " + path)
    for path, expected in plan["checkpoint_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("Checkpoint changed: " + path)
    if sha(plan["protocol"]) != plan["protocol_sha256"]:
        raise RuntimeError("Protocol changed")


def verify_inputs(folder, expected):
    if {p.name: sha(p) for p in Path(folder).glob("*.png")} != expected:
        raise RuntimeError("Canonical inputs changed: " + str(folder))


def stage(root, name, command, plan, state):
    check_sources(plan)
    state.update(status="running", current=name)
    save(root / "session.json", state)
    print("Starting " + name + " at " + datetime.now().astimezone().isoformat(), flush=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    with (root / (name + ".log")).open("x") as log:
        process = subprocess.Popen(command, cwd=plan["working_directory"], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            previous, last_print = None, time.monotonic()
            while process.poll() is None:
                time.sleep(2)
                counts = {str(p.parent.relative_to(root)): len(p.read_text().splitlines())
                          for p in root.rglob("records.jsonl")}
                generated = root / "data/gen512"
                if generated.exists():
                    counts["generated"] = len(list(generated.glob("*.png")))
                if counts != previous or time.monotonic() - last_print >= 30:
                    print("  " + name + ": " + json.dumps(counts), flush=True)
                    previous, last_print = counts, time.monotonic()
            if process.returncode:
                raise RuntimeError(f"{name} failed ({process.returncode}); see {root / (name + '.log')}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    state["completed"].append(name)
    save(root / "session.json", state)


def evaluation_command(python, device, real, out, methods, full=False):
    return python + ["m2_evaluate.py", "--device", device, "--real_dir", str(real), "--out", str(out),
                     "--detectors", "E,R,D,S" if full else "E", "--conditions", "none,jpeg75,resize05" if full else "none",
                     "--runs", *[name + "=" + str(folder) for name, folder in methods.items()]]


def development_metrics(evaluation, methods, inputs, zero):
    with (evaluation / "per_image.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    canonical = {f"{i:04d}": np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
                 for i, p in enumerate(sorted(inputs.glob("*.png")))}
    for image_id in canonical:
        if sha(methods["OLD_RECON8"] / f"{image_id}_adv_image.png") != sha(zero / f"{image_id}_adv_image.png"):
            raise RuntimeError("Old/new zero-step reconstructions differ: " + image_id)
    summaries = {}
    for name, folder in methods.items():
        values = [r for r in rows if r["run"] == name and r["detector"] == "E" and r["condition"] == "none"]
        if len(values) != 10 or {r["image_id"] for r in values} != set(canonical):
            raise RuntimeError("Incomplete development cell " + name)
        for row in values:
            image_id = row["image_id"]
            clean = np.asarray(Image.open(folder / f"{image_id}_originImage.png").convert("RGB"), dtype=np.uint8)
            output = np.asarray(Image.open(folder / f"{image_id}_adv_image.png").convert("RGB"), dtype=np.uint8)
            if not np.array_equal(clean, canonical[image_id]):
                raise RuntimeError("Development source differs: " + name + "/" + image_id)
            if np.abs(output.astype(np.int16) - clean.astype(np.int16)).max() > 8:
                raise RuntimeError("Development budget exceeded: " + name + "/" + image_id)
            if name.startswith("Q") and sha(folder / f"{image_id}_reset_image.png") != sha(zero / f"{image_id}_adv_image.png"):
                raise RuntimeError("Development reset differs: " + name + "/" + image_id)
        eligible = [r for r in values if r["clean_pred"] == "0"]
        lpips = [float(r["lpips"]) for r in values]
        records = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
        qualified = lambda r: float(r["lpips"]) <= .03 and float(r["psnr"]) >= 30
        summaries[name] = {"n": 10, "n_clean_detected": len(eligible),
                           "classifier_success": sum(r["adv_pred"] == "1" for r in eligible),
                           "low_distortion_success": sum(r["adv_pred"] == "1" and qualified(r) for r in eligible),
                           "quality_qualified": sum(qualified(r) for r in values),
                           "mean_lpips": statistics.mean(lpips), "max_lpips": max(lpips),
                           "mean_lpips_pass": statistics.mean(lpips) <= .015,
                           "actual_gradient_steps": sum(int(r.get("actual_gradient_steps", len(r.get("history", [])))) for r in records),
                           "output_sha256": {image_id: sha(folder / f"{image_id}_adv_image.png") for image_id in canonical}}
    clean_sets = [{r["image_id"] for r in rows if r["run"] == name and r["clean_pred"] == "0"} for name in methods]
    if any(value != clean_sets[0] for value in clean_sets[1:]):
        raise RuntimeError("Development eligibility differs between methods")
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["dev", "validate"], required=True)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--candidate_freeze", type=Path)
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parent
    protocol = here.parent / "docs/M3_LAO_V2_PROTOCOL_ZH.md"
    archive = here.parents[2] / "snapshots" / SNAPSHOT
    archive_meta = json.loads((archive / "snapshot.json").read_text())
    if archive_meta.get("status") != "verified":
        raise RuntimeError("Pre-development snapshot is not verified")
    first = here / "results/m3_lao_validation" / FIRST_RUN
    first_plan = json.loads((first / "plan.json").read_text())
    development_reference = json.loads((here / "results/m3_lao_dev" / DEV_REFERENCE / "plan.json").read_text())
    diffusion_model = development_reference["diffusion_model"]
    for relative, expected in diffusion_model["file_sha256"].items():
        if sha(Path(diffusion_model["resolved_local_path"]) / relative) != expected:
            raise RuntimeError("Frozen SD2.1 model changed: " + relative)
    for name, expected in first_plan["source_sha256"].items():
        if sha(here / name) != expected:
            raise RuntimeError("First-round implementation changed: " + name)
    python = [sys.executable, "-B", "-u"]
    parent = here / "results/m3_lao_v2"
    parent.mkdir(parents=True, exist_ok=True)
    if args.root is None:
        root = Path(tempfile.mkdtemp(prefix=args.phase + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_", dir=parent))
    else:
        root = args.root.resolve()
        root.mkdir(parents=True, exist_ok=False)
    sources = {str(p): sha(p) for p in source_files(here)}
    checkpoint_hashes = {str(here / "checkpoints" / n): h for n, h in first_plan["checkpoint_sha256"].items()}
    plan = {"created_at": datetime.now().astimezone().isoformat(), "phase": args.phase,
            "working_directory": str(here), "device": args.device, "snapshot": str(archive),
            "source_sha256": sources, "checkpoint_sha256": checkpoint_hashes,
            "diffusion_model": diffusion_model,
            "protocol": str(protocol), "protocol_sha256": sha(protocol), "model_training": False,
            "budget_pixels": 8, "quality_screen": {"psnr_min": 30.0, "lpips_max": .03},
            "legacy_quality_screen": {"psnr_min": 30.0, "lpips_max": .15}, "new_mean_lpips_max": .015}
    frozen = root / "frozen_code"
    for path in sources:
        destination = frozen / Path(path).relative_to(here)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(protocol, root / "protocol.md")
    state = {"status": "planned", "completed": [], "created_at": plan["created_at"]}
    base_model = first_plan["attack_model"]["path"]
    common = ["--device", args.device, "--pretrained_diffusion_path", base_model]
    try:
        if args.phase == "dev":
            inputs, real = here / "data15/gen/dev", here / "data15/real/dev"
            reference = here / "results/m3_lao_dev" / DEV_REFERENCE
            old = root / "old"
            new = root / "new"
            methods = {"OLD_RECON8": old / "RECON8", "OLD_LAO8": old / "LAO8", "NEW_RECON8": new / "NEW_RECON8",
                       "Q10": new / "Q10", "Q40": new / "Q40", "PGD10": root / "pgd/PGD10", "PGD10_MATCHED": root / "pgd/PGD10_MATCHED"}
            plan.update(n_generated=10, n_real=10, method_paths={k: str(v) for k, v in methods.items()},
                        input_sha256={p.name: sha(p) for p in inputs.glob("*.png")},
                        real_sha256={p.name: sha(p) for p in real.glob("*.png")},
                        candidates=[{"name": "Q10", "max_iterations": 10, "lr": .01},
                                    {"name": "Q40", "max_iterations": 40, "lr": .01},
                                    {"name": "Q40_LR02", "max_iterations": 40, "lr": .02}],
                        optional_third_rule="run iff Q40 low-distortion successes < clean-detected count OR mean LPIPS > .015")
            commands = {
                "old_export": python + ["m3_lao_budget.py", "--source_run", str(reference / "G1"), "--baseline_run", str(reference / "G0"),
                                         "--images_root", str(inputs), "--out_root", str(old), "--budget_pixels", "8"],
                "new_zero": python + ["m3_lao_v2.py", *common, "--images_root", str(inputs), "--out_root", str(new),
                                       "--max_iterations", "0", "--name", "NEW_RECON8"],
                "q40": python + ["m3_lao_v2.py", *common, "--images_root", str(inputs), "--out_root", str(new),
                                  "--max_iterations", "40", "--lr", ".01", "--name", "Q40", "--prefix_name", "Q10"],
                "pgd": python + ["m3_lao_pgd_v2.py", "--device", args.device, "--images_root", str(inputs), "--out_root", str(root / "pgd")],
                "eval_first": evaluation_command(python, args.device, real, root / "eval_first", methods),
                "q40_lr02": python + ["m3_lao_v2.py", *common, "--images_root", str(inputs), "--out_root", str(new),
                                       "--max_iterations", "40", "--lr", ".02", "--name", "Q40_LR02"],
                "eval_final": evaluation_command(python, args.device, real, root / "eval_final", {**methods, "Q40_LR02": new / "Q40_LR02"}),
            }
            plan["commands"] = commands
            save(root / "plan.json", plan)
            for name in ["old_export", "new_zero", "q40", "pgd", "eval_first"]:
                verify_inputs(inputs, plan["input_sha256"])
                verify_inputs(real, plan["real_sha256"])
                stage(root, name, commands[name], plan, state)
            metrics = development_metrics(root / "eval_first", methods, inputs, new / "NEW_RECON8")
            q40 = metrics["Q40"]
            third = q40["low_distortion_success"] < q40["n_clean_detected"] or not q40["mean_lpips_pass"]
            save(root / "optional_third_decision.json", {"run_third": third, "Q40": q40})
            evaluation = root / "eval_first"
            if third:
                for name in ["q40_lr02", "eval_final"]:
                    verify_inputs(inputs, plan["input_sha256"])
                    stage(root, name, commands[name], plan, state)
                methods["Q40_LR02"] = new / "Q40_LR02"
                evaluation = root / "eval_final"
                metrics = development_metrics(evaluation, methods, inputs, new / "NEW_RECON8")
            candidates = [c for c in plan["candidates"] if c["name"] in metrics]
            admissible = [c for c in candidates if metrics[c["name"]]["mean_lpips_pass"]]
            pool = admissible or candidates
            chosen = min(pool, key=lambda c: (-metrics[c["name"]]["low_distortion_success"], metrics[c["name"]]["mean_lpips"],
                                              metrics[c["name"]]["actual_gradient_steps"], plan["candidates"].index(c)))
            selected = metrics[chosen["name"]]
            improved = selected["mean_lpips_pass"] and selected["low_distortion_success"] > metrics["OLD_LAO8"]["low_distortion_success"]
            evidence = {"created_at": datetime.now().astimezone().isoformat(), "status": "verified",
                        "split": "old ten-image development set only", "selected_configuration": chosen,
                        "selection_rule": "mean LPIPS <= .015, then most low-distortion successes, least mean LPIPS, fewest gradient updates, lowest candidate index; if none admissible, report no development improvement and freeze best tested candidate",
                        "development_improvement_observed": improved, "mean_guard_admissible_candidate_exists": bool(admissible),
                        "metrics": metrics, "evaluation_csv": str(evaluation / "per_image.csv"),
                        "evaluation_sha256": sha(evaluation / "per_image.csv"), "method_paths": {k: str(v) for k, v in methods.items()}}
            save(root / "selection.json", evidence)
            check_sources(plan)
            freeze = {"status": "frozen", "created_at": datetime.now().astimezone().isoformat(),
                      "selected_configuration": {**chosen, "iterations": chosen["max_iterations"], "candidate_name": chosen["name"],
                                                 "budget_pixels": 8, "lpips_cap": .03, "margin_buffer": .1,
                                                 "refinement_steps": 5, "kappa": 3, "lpips_weight": 1, "control_scale": .5},
                      "development_evidence": {"path": str(root / "selection.json"), "sha256": sha(root / "selection.json")},
                      "development_improvement_observed": improved, "source_sha256": sources,
                      "protocol": str(protocol), "protocol_sha256": sha(protocol),
                      "checkpoint_sha256": checkpoint_hashes, "snapshot": str(archive)}
            freeze["diffusion_model"] = diffusion_model
            save(root / "freeze.json", freeze)
            print("Frozen candidate: " + str(root / "freeze.json"), flush=True)
        else:
            if args.candidate_freeze is None:
                raise ValueError("validation requires --candidate_freeze")
            freeze_path = args.candidate_freeze.resolve()
            freeze = json.loads(freeze_path.read_text())
            if (freeze.get("status") != "frozen" or freeze["source_sha256"] != sources
                    or freeze["protocol_sha256"] != sha(protocol) or freeze["diffusion_model"] != diffusion_model):
                raise RuntimeError("Candidate code/protocol changed after development freeze")
            check_sources({**plan, "source_sha256": freeze["source_sha256"]})
            chosen = freeze["selected_configuration"]
            inputs, real = root / "data/gen/test", root / "data/real/test"
            methods = {"OLD_RECON8": root / "old/RECON8", "OLD_LAO8": root / "old/LAO8",
                       "NEW_RECON8": root / "new/NEW_RECON8", "NEW_LAO8": root / "new/NEW_LAO8",
                       "PGD10": root / "pgd/PGD10", "PGD10_MATCHED": root / "pgd/PGD10_MATCHED"}
            plan.update(n_generated=50, n_real=50, method_paths={k: str(v) for k, v in methods.items()},
                        candidate_freeze=str(freeze_path), candidate_freeze_sha256=sha(freeze_path),
                        primary={"run": "NEW_LAO8", "comparator": "OLD_LAO8", "detector": "E", "condition": "none",
                                 "bootstrap_seed": 20261018, "bootstrap_resamples": 10000})
            commands = {"data": python + ["m3_lao_newdata_v2.py", "--out", str(root / "data"), "--candidate_freeze", str(freeze_path),
                                           "--model_path", first_plan["generation"]["generator_path"], "--device", args.device, "--n", "50"]}
            for name in ["RAW_RECON", "RAW_LAO"]:
                command = list(first_plan["commands"][name])
                command[0] = sys.executable
                command[command.index("--images_root") + 1] = str(inputs)
                command[command.index("--save_dir") + 1] = str(root / name)
                command[command.index("--device") + 1] = args.device
                commands[name] = command
            commands.update({
                "old_export": python + ["m3_lao_budget.py", "--source_run", str(root / "RAW_LAO"), "--baseline_run", str(root / "RAW_RECON"),
                                         "--images_root", str(inputs), "--out_root", str(root / "old"), "--budget_pixels", "8"],
                "new_zero": python + ["m3_lao_v2.py", *common, "--images_root", str(inputs), "--out_root", str(root / "new"),
                                       "--max_iterations", "0", "--lr", str(chosen["lr"]), "--name", "NEW_RECON8"],
                "new_lao": python + ["m3_lao_v2.py", *common, "--images_root", str(inputs), "--out_root", str(root / "new"),
                                      "--max_iterations", str(chosen["max_iterations"]), "--lr", str(chosen["lr"]), "--name", "NEW_LAO8"],
                "pgd": python + ["m3_lao_pgd_v2.py", "--device", args.device, "--images_root", str(inputs), "--out_root", str(root / "pgd")],
                "evaluation": evaluation_command(python, args.device, real, root / "eval", methods, full=True),
                "report": python + ["m3_lao_v2_report.py", "--root", str(root), "--out", str(root / "report")],
            })
            plan["commands"] = commands
            save(root / "plan.json", plan)
            canonical = None
            for name, command in commands.items():
                if canonical:
                    verify_inputs(inputs, canonical["generated"])
                    verify_inputs(real, canonical["real"])
                stage(root, name, command, plan, state)
                if name == "data":
                    canonical = {"generated": {p.name: sha(p) for p in inputs.glob("*.png")},
                                 "real": {p.name: sha(p) for p in real.glob("*.png")}}
                    if len(canonical["generated"]) != 50 or len(canonical["real"]) != 50:
                        raise RuntimeError("Incomplete second-batch data")
        state.update(status="complete", finished_at=datetime.now().astimezone().isoformat())
        save(root / "session.json", state)
        print("Complete: " + str(root), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), finished_at=datetime.now().astimezone().isoformat())
        save(root / "session.json", state)
        raise


if __name__ == "__main__":
    main()
