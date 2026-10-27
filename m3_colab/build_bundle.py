"""Build the course-only M3 resources from an immutable, verified snapshot.

This is a packaging command, not an attack or training command. Run with --help.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


CORE_FILES = [
    "m2_stealth.py", "m2_evaluate.py", "m2_device.py", "other_attacks.py",
    "diff_latent_attack.py", "ControlVAE.py", "network.py", "utils.py",
    "distances.py", "m2_data.py", "README.md",
]
CHECKPOINTS = ["Controlvae.pt", "efficientnet-b0.pth", "resnet50.pth", "deit.pth", "swin-t.pth"]
SD_REPO = "Manojb/stable-diffusion-2-1-base"
SD_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def entry(path, relative):
    return {"path": str(relative), "sha256": digest(path), "bytes": path.stat().st_size}


def package(snapshot, output, sd_model, weights=True):
    source = snapshot / "repository" / "StealthDiffusion"
    records = json.loads((snapshot / "SHA256SUMS.json").read_text())
    known = {row["path"]: row["sha256"] for row in records}
    bundle = output / "course_bundle"
    if bundle.exists():
        raise FileExistsError(f"Refusing to overwrite {bundle}")
    bundle.mkdir(parents=True)
    provenance = []

    def archived_copy(src, rel):
        expected = known[str(src.relative_to(snapshot))]
        assert digest(src) == expected, f"Snapshot hash mismatch: {src.name}"
        dest = bundle / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        provenance.append({"path": str(rel), "snapshot_path": str(src.relative_to(snapshot)),
                           "source_sha256": expected})

    for filename in CORE_FILES:
        archived_copy(source / filename, Path("src") / filename)
    for group in ("gen", "real"):
        for split in ("dev", "test"):
            for path in sorted((source / "data15" / group / split).iterdir()):
                if path.suffix in (".png", ".csv"):
                    archived_copy(path, Path("data") / group / split / path.name)
    for filename in ("summary.csv", "per_image.csv", "real_predictions.csv"):
        archived_copy(source / "results" / "m3_sd15" / "eval" / filename,
                      Path("historical") / filename)

    # Portability only: all detector parameters are subsequently loaded strictly.
    path = bundle / "src" / "other_attacks.py"
    text = path.read_text()
    changes = [
        ("os.environ['CUDA_VISIBLE_DEVICES'] = \"1\"", "os.environ['CUDA_VISIBLE_DEVICES'] = \"0\""),
        ('EfficientNet.from_pretrained("efficientnet-b0", num_classes=1, image_size=None, )',
         'EfficientNet.from_name("efficientnet-b0", num_classes=1, image_size=None, )'),
        ("timm.create_model('swin_base_patch4_window7_224', pretrained=True)",
         "timm.create_model('swin_base_patch4_window7_224', pretrained=False)"),
    ]
    for old, new in changes:
        assert text.count(old) == 1, f"Unexpected source: {old}"
        text = text.replace(old, new)
    path.write_text(text)
    (bundle / "PORTABILITY.md").write_text(
        "# Course snapshot and portability\n\n"
        f"Source snapshot: `{snapshot.name}`. Original files were checked against its SHA-256 manifest.\n\n"
        "The only attack-source edit is in `src/other_attacks.py`: use visible GPU 0; "
        "construct EfficientNet and Swin without downloading generic ImageNet initialization. "
        "The supplied forensic checkpoint is still loaded with strict=True (PyTorch default). "
        "No attack objective, gradient loop, detector checkpoint, or evaluation rule is changed.\n\n"
        "The portable runner creates fresh directories and calls the archived commands explicitly. "
        "It never imports the later m3_lao_support or LAO v1/v2/v3 experiments.\n\n"
        "Image manifests identify the fixed SD1.5-generated inputs and Imagenette real inputs. "
        "The 512-pixel generation cache is not needed for this saved-224-pixel-input reproduction.\n\n"
        "Historical CSVs are archived measurements, not newly executed model predictions. "
        "The notebook labels its fresh small execution separately.\n"
    )
    peer = bundle / "peer"
    peer.mkdir()
    write_json(peer / "summary.json", {
        "status": "pending", "review_status": "pending", "attempted": 7,
        "detector_flips": 7, "control_rows": 3, "source_samples": 10,
        "eligible_attempts": 7, "confirmed_count": None, "review_complete": False,
        "norm": "raw latent L-infinity", "budget": 0.1,
        "measurement": "Saved FP16 values cast to FP32; zero tolerance",
        "batch": "One shuffled batch: seven early-stopped PGD outputs and three unmodified controls",
        "scope": "Separate exploratory latent-access track; not the original prompt-only challenge",
        "sample_limit": "Supplied historical training-positive examples; not held-out evidence. No FPR, AUROC or population-generalization claim.",
        "success_rule": "Usable decoded image AND human-confirmed visible physical violence AND frozen CNN predicts benign",
        "disclosure": "Individual scores, sample identities and control mapping withheld until blind review completes",
        "provenance": "Archived pilot_blind_7plus3_2026-10-18/verification/summary.json; aggregate status only",
        "selection_note": "Seven perturbed outputs were retained from an earlier local eight-input execution; the submitted batch is not a pre-registered seven-input test. No extra peer batch is run here.",
        "peer_thread": "https://edstem.org/au/courses/37864/discussion/3533574?comment=7924177",
    })
    resources = {"weights": [], "sd_model": {"repo_id": SD_REPO, "revision": SD_REVISION, "files": []}}
    weight_sources = [(source / "checkpoints" / name, Path("checkpoints") / name) for name in CHECKPOINTS]
    weight_sources.append((source / ".cache" / "checkpoints" / "vgg16-397923af.pth",
                           Path(".cache/checkpoints/vgg16-397923af.pth")))
    for src, rel in weight_sources:
        assert digest(src) == known[str(src.relative_to(snapshot))]
        resources["weights"].append(entry(src, rel))
    for src in sorted(sd_model.rglob("*")):
        if src.is_file():
            resources["sd_model"]["files"].append(entry(src, src.relative_to(sd_model)))
    write_json(bundle / "resources.json", resources)
    write_json(bundle / "source_provenance.json", provenance)
    if weights:
        with zipfile.ZipFile(output / "m3_weights.zip", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for src, rel in weight_sources:
                z.write(src, str(rel))
        print("Weights packed:", (output / "m3_weights.zip").stat().st_size, "bytes", flush=True)
    finalize(bundle, output)
    return bundle


def finalize(bundle, output):
    # This can be rerun after report/runtime helpers have been completed.
    helpers = Path(__file__).parent
    for filename in ("m3_runtime.py", "m3_reporting.py"):
        if (helpers / filename).exists():
            shutil.copyfile(helpers / filename, bundle / filename)
    files = [entry(p, p.relative_to(bundle)) for p in sorted(bundle.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts]
    write_json(bundle / "manifest.json", {
        "version": 1,
        "source_snapshot": "before_lao_architecture_optimization_20261018_145942",
        "purpose": "M3 course reproduction; later LAO research excluded",
        "files": files,
    })
    with zipfile.ZipFile(output / "m3_course_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for row in files:
            z.write(bundle / row["path"], row["path"])
        z.write(bundle / "manifest.json", "manifest.json")
    print("Course bundle:", len(files), "files,", (output / "m3_course_bundle.zip").stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--sd-model", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--finalize", action="store_true")
    args = p.parse_args()
    if args.finalize:
        finalize(args.output / "course_bundle", args.output)
    else:
        assert args.snapshot and args.sd_model
        package(args.snapshot, args.output, args.sd_model)
