"""Prepare and verify the pilot; perturbation runs are locked pending agreement.

Run from the repository root: python -m peer_eval.predecodeguard.run --help
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from .adapter import FrozenGuard
from .contract import (HERE, METHODS, check_protocol, load_latent, new_output,
                       package_manifest, proposal_digest, read_json, require_agreement,
                       sha256, write_json)
from .evaluate import evaluate_batch, official_verify
from .report import summarize, write_reports


def check_output_location(out: Path, package: Path):
    out, package = out.resolve(), package.resolve()
    if out == package or package in out.parents or out in package.parents:
        raise ValueError("Outputs must be outside the immutable supplied package")


def provenance(package, protocol, command, method):
    sources = {str(p.relative_to(HERE)):sha256(p) for p in sorted(HERE.glob("*.py"))}
    sources["package_lock.json"] = sha256(HERE / "package_lock.json")
    return {"started_at":datetime.now(timezone.utc).isoformat(), "command":command,
            "method":method, "source_package":str(package),
            "proposal_sha256":proposal_digest(protocol), "protocol":protocol,
            "migration_code_sha256":sources,
            "environment":{"python":sys.version, "torch":torch.__version__, "numpy":np.__version__,
                           "platform":platform.platform(), "device":"cpu", "torch_threads":4},
            "formal_perturbations_executed":command == "experiment", "status":"started"}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("baseline", "control", "plan", "experiment", "audit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--package", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten")
        cmd.add_argument("--protocol", type=Path, default=HERE / "protocol.pending.json")
        if name == "control":
            cmd.add_argument("--kind", choices=("copy", "roundtrip"), required=True)
        if name == "experiment":
            cmd.add_argument("--method", choices=METHODS, required=True)
            cmd.add_argument("--execute-approved", action="store_true")
        if name == "audit":
            cmd.add_argument("--method", choices=METHODS, required=True)
            cmd.add_argument("--returned", type=Path, required=True)
            cmd.add_argument("--review", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    package = args.package.resolve()
    protocol = read_json(args.protocol)
    # Validate protocol/gate before model loading, output mkdir or source arrays.
    manifest = package_manifest(package)
    check_protocol(protocol, manifest, sha256(package / "manifest.json"))
    if args.command == "experiment":
        require_agreement(protocol, args.method, execute_approved=args.execute_approved,
                          protocol_path=args.protocol)
    check_output_location(args.out, package)
    if args.command == "audit":
        ret = args.returned.resolve()
        out_abs = args.out.resolve()
        if ret == out_abs or ret in out_abs.parents or out_abs in ret.parents:
            raise ValueError("Audit output must be separate from returned files")
    out = new_output(args.out)
    method = getattr(args, "method", getattr(args, "kind", args.command))
    meta = provenance(package, protocol, args.command, method)
    write_json(out / "run_config.json", meta)
    try:
        if args.command == "plan":
            plan = {"status":"prepared_not_executed", "formal_perturbations_executed":False,
                    "protocol_status":protocol["status"], "proposal_sha256":proposal_digest(protocol),
                    "independent_source_samples":len(manifest["samples"]),
                    "requested_return_batches":protocol["returned_batches"],
                    "requested_return_files":protocol["returned_batches"] * len(manifest["samples"]),
                    "requested_review_images":(protocol["returned_batches"] + 1) * len(manifest["samples"]),
                    "experiments":[]}
            for name in protocol["requested_methods"]:
                config = protocol["methods"][name]
                plan["experiments"].append({"method":name, **config,
                    "attempted_samples":len(manifest["samples"]) - len(config["control_filenames"]),
                    "epsilon":protocol["budget"]["epsilon"]})
            write_json(out / "experiment_plan.json", plan)
            with (out / "planned_samples.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["method", "filename", "original_sha256", "attempted", "seed", "status"])
                writer.writeheader()
                for arm in plan["experiments"]:
                    for sample_index, sample in enumerate(manifest["samples"]):
                        writer.writerow({"method":arm["method"], "filename":sample["filename"],
                                         "original_sha256":sample["sha256"],
                                         "attempted":sample["filename"] not in arm["control_filenames"],
                                         "seed":arm["seed"] + sample_index,
                                         "status":"planned_not_executed"})
            meta["status"] = "prepared"
            write_json(out / "run_config.json", meta)
            print(json.dumps(plan, ensure_ascii=False))
            return 0

        guard = FrozenGuard(package)
        records, attempts = {}, set()
        epsilon = protocol["budget"]["epsilon"]
        returned = package / "latents"
        if args.command in ("control", "experiment"):
            returned = out / "returned"
            returned.mkdir()
            if args.command == "experiment":
                from .methods import generate
                cfg = protocol["methods"][method]
                attempts = {r["filename"] for r in manifest["samples"]} - set(cfg["control_filenames"])
                (out / "traces").mkdir()
            for sample_index, sample in enumerate(manifest["samples"]):
                name = sample["filename"]
                start = time.monotonic()
                try:
                    src = package / "latents" / name
                    if args.command == "control" and args.kind == "copy" or args.command == "experiment" and name not in attempts:
                        shutil.copy2(src, returned / name)
                        info = {"steps":0}
                    else:
                        original = load_latent(src)
                        if args.command == "control":
                            array = original.astype(np.float32).astype(np.float16)
                            info = {"steps":0}
                        else:
                            array, info = generate(original, guard.logits, method=method, epsilon=epsilon,
                                                   steps=cfg["steps"], alpha=cfg["alpha"], seed=cfg["seed"] + sample_index,
                                                   threshold=guard.threshold, score_fn=guard.score,
                                                   early_stop_margin=cfg.get("early_stop_margin", 0.0))
                            write_json(out / "traces" / (Path(name).stem + ".json"), info)
                        np.save(returned / name, array, allow_pickle=False)
                    records[name] = {"steps":info["steps"], "seconds":time.monotonic() - start}
                except Exception as exc:
                    # Keep the sample in the evaluation denominator, with its failure.
                    records[name] = {"steps":None, "seconds":time.monotonic() - start,
                                     "error":f"{type(exc).__name__}: {exc}"}
                write_json(out / "execution_records.json", records)
        elif args.command == "audit":
            returned = args.returned.resolve()
            cfg = protocol["methods"][method]
            attempts = {r["filename"] for r in manifest["samples"]} - set(cfg["control_filenames"])

        rows, audit = evaluate_batch(package, returned, guard, method=method,
                                     attempt_names=attempts, epsilon=epsilon, records=records)
        review = None
        if args.command == "audit" and args.review:
            with args.review.open(newline="", encoding="utf-8") as f:
                review = list(csv.DictReader(f))
        summary = summarize(rows, review)
        summary["artifact_audit"] = audit
        summary["protocol_status"] = protocol["status"]
        summary["formal_perturbations_executed"] = args.command == "experiment"
        summary["source_exposure"] = "historical_training_examples_exploratory_only"
        write_reports(out, rows, summary)
        if audit["complete"] and all(r["valid"] for r in rows):
            try:
                summary["official_verification"] = official_verify(package, out, rows,
                    None if args.command == "baseline" else returned)
            except Exception as exc:
                summary["official_verification"] = {"status":"failed", "error":str(exc)}
                write_reports(out, rows, summary)
                raise
            if args.command in ("baseline", "control") and any(not r["identical"] or r["linf"] != 0 or r["flagged"] != r["baseline_flagged"] for r in rows):
                raise ValueError("Preparation control changed values or decisions")
            meta["status"] = "verified"
        else:
            summary["official_verification"] = {"status":"not_run_invalid_batch"}
            meta["status"] = "invalid_batch"
        write_reports(out, rows, summary)
        write_json(out / "run_config.json", meta)
        print(json.dumps({"status":meta["status"], "samples":len(rows), "out":str(out.resolve()),
                          "formal_perturbations_executed":meta["formal_perturbations_executed"]}))
        return 0 if meta["status"] == "verified" else 1
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "run_config.json", meta)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, PermissionError, FileExistsError, FileNotFoundError) as exc:
        print(f"Blocked/failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
