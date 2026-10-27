"""Create the M3 notebook source. Execute it separately to obtain genuine outputs."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import textwrap
import nbformat as nbf


def create(delivery: Path, output: Path):
    sha = hashlib.sha256((delivery / "m3_course_bundle.zip").read_bytes()).hexdigest()
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # Dark.ModelBreakers · Milestone 3
    **Saved-image evasion, score capping, and robustness under image transformations**  
    FIT5230 · LIU JUNPENG and WANG YUNJIE · course version frozen 18 October 2026

    This notebook reproduces the **course implementation frozen before the later LAO architecture work**.
    It runs a small, fixed example end to end and separately audits the archived 100-image course results.
    The saved execution identifies its actual hardware. A local Apple MPS execution is **not** a claim that this notebook has already run on Colab CUDA.

    **PreDecodeGuard human review: PENDING.** Classifier flips are candidate outcomes, not confirmed content-preserving bypasses.
    Liu Junpeng's additional individual peer strategy is still being developed; it is not presented as a completed experiment here.
    Later LAO v1/v2/v3 architecture experiments and paper extensions are outside this notebook.

    ## Run instructions
    1. Select a GPU runtime. This notebook has completed on **NVIDIA L4 (22 GiB, CUDA, FP32)** with the default 3-image workload. The upstream recommendation is at least 32 GB, so A100 offers more headroom. A 16 GB T4 has not been verified.
    2. Use the prepared Google Drive folder `FIT5230_M3` (or change `RESOURCE_FOLDER` below). It contains `m3_course_bundle.zip` and a `weights/` folder with five author checkpoints and the LPIPS VGG cache. Alternatively, upload the delivered `m3_weights.zip` into the same folder.
    3. Run all cells. Authorize your own Drive mount when prompted. The attack model is fetched from a pinned HF revision, then verified against file hashes.
    4. Keep the cell outputs when saving. The last cell also saves a ZIP of new images, logs, metrics, and figures back to Drive.

    Default: **3 fixed generated inputs × 7 methods**, 10 real development images for CAP calibration, and 100 separate real test images for FPR.
    Set `N_IMAGES=100` to rerun the course-sized attack evaluation. The small default proves execution, not statistical generalization.
    No model is retrained. Fresh output folders prevent accidental reuse of archived results.
    """)
    md("""
    ## 1. Question and experimental design
    Our M2 work extended StealthDiffusion's PGD → latent optimization → Control-VAE pipeline with ensemble surrogates and loss-path options.
    M3 asks what actually affects the **saved artifact**, whether very high fixed-threshold evasion hides score overshoot, and whether evasion survives JPEG/resizing.

    | Configuration | What changes | What it tests |
    |---|---|---|
    | A5 | Archived released-pipeline baseline | Reference behavior |
    | A2 | Zero latent-optimization iterations | Contribution of the existing latent loop |
    | X2 | Feed the clean image into the Control-VAE control path | Dependence on the control image |
    | X4 | Clean inversion, clean control, zero latent iterations | Reconstruction-only control |
    | ENS | Optimize E/R/D instead of E alone | Within-collection surrogate transfer |
    | CAPB_ENS | Capped PGD + crossing-step bisection, E/R/D | Overshoot and perceptual distortion |
    | EOT_ENS | Average PGD loss over identity/JPEG75/resize0.5 | Transformation robustness |

    **CAPB and EOT are separate experiments. No CAP+EOT combination is claimed.**
    E=EfficientNet-B0, R=ResNet-50, D=DeiT, S=Swin. The supplied file is named `swin-t.pth`, but the wrapper instantiates Swin-base.
    S is excluded from the optimization loss. It is from the same supplied detector collection, so this is limited transfer evidence.

    The course data consist of 100 fixed SD1.5-generated test images and 100 Imagenette real test images, with separate 10-image dev splits.
    The attack uses the archived SD2.1 backbone. Generation is already complete: manifests preserve prompts/seeds/model provenance.
    CAP calibration reads **real/dev only**. Test outputs are evaluated after saving and reloading PNGs in a separate process.
    This is reproduction of the existing course test set, not a newly held-out validation study.

    Conditional ASR = originally detected AI images that become real / originally detected AI images, under the **same** transform.
    Always show the numerator/denominator and real-image FPR. The decoder changes the image beyond the PGD stage;
    this course pipeline does **not** impose a common final-image L∞ bound across all methods.
    """)
    code("""
    # Configuration: change only these inputs for another supported runtime.
    import os, sys, json, hashlib, shutil, subprocess, zipfile, uuid, datetime
    from pathlib import Path
    from IPython.display import display, Markdown
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    RESOURCE_FOLDER = "/content/drive/MyDrive/FIT5230_M3"
    N_IMAGES = 3
    METHODS = ("A5", "A2", "X2", "X4", "ENS", "CAPB_ENS", "EOT_ENS")
    CONDITIONS = ("none", "jpeg75", "resize05")
    RUN_ID = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    RESOURCE = Path(RESOURCE_FOLDER) if IN_COLAB else Path(os.environ.get("M3_RESOURCE_DIR", ".")).resolve()
    WORK = (Path("/content/m3_runs") if IN_COLAB else Path(os.environ.get("M3_WORK_ROOT", "m3_runs"))) / RUN_ID
    WORK.mkdir(parents=True, exist_ok=False)
    print("Execution mode:", "Colab CUDA" if IN_COLAB else "Local Jupyter; hardware is recorded below")
    print("Run ID:", RUN_ID, "| generated inputs per method:", N_IMAGES)
    print("Methods:", ", ".join(METHODS))
    """)
    code(f"""
    # Verify the delivered ZIP before loading its code. Never extract arbitrary paths.
    EXPECTED_BUNDLE_SHA256 = "{sha}"
    if IN_COLAB:
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
    def file_sha256(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    def extract_checked(archive, destination):
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = destination / entry.filename
                assert not Path(entry.filename).is_absolute() and ".." not in Path(entry.filename).parts
                assert target.resolve().is_relative_to(destination)
                assert (entry.external_attr >> 16) & 0o170000 != 0o120000, "ZIP symlinks are unsupported"
            z.extractall(destination)
    archive = RESOURCE / "m3_course_bundle.zip"
    assert archive.is_file(), "Place m3_course_bundle.zip in RESOURCE_FOLDER first."
    assert file_sha256(archive) == EXPECTED_BUNDLE_SHA256, "Course bundle changed; use the matching notebook and ZIP."
    BUNDLE = WORK / "bundle"
    extract_checked(archive, BUNDLE)
    sys.path.insert(0, str(BUNDLE))
    sys.dont_write_bytecode = True
    import m3_runtime as runtime
    import m3_reporting as report
    bundle_check = runtime.verify_bundle(BUNDLE)
    print("Verified source snapshot:", bundle_check["source_snapshot"])
    print("Verified files:", bundle_check["verified_files"], "| image counts:", bundle_check["image_counts"])
    print("Dev/test file-hash overlap:", bundle_check["dev_test_hash_overlap"])
    """)
    code("""
    # A separate Python 3.10 process isolates the archived ML stack from Colab's kernel.
    if IN_COLAB:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.9.4"], check=True)
        UV = shutil.which("uv")
        PYTHON = "/content/m3_sdenv/bin/python"
        if not Path(PYTHON).exists():
            subprocess.run([UV, "venv", "--python", "3.10", "/content/m3_sdenv"], check=True)
        subprocess.run([UV, "pip", "install", "--python", PYTHON, "torch==2.2.2", "torchvision==0.17.2",
                        "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
        PACKAGES = ["setuptools==80.9.0", "numpy==1.26.4", "diffusers==0.12.1", "huggingface_hub==0.23.4",
                    "transformers==4.30.2", "accelerate==0.20.3", "timm==0.6.12", "efficientnet-pytorch==0.7.1",
                    "torchattacks==3.5.1", "lossers==0.0.9", "eagerpy==0.30.0", "natsort==8.4.0",
                    "tensorboardX==2.6.2.2", "scipy==1.11.4", "scikit-learn==1.3.2",
                    "opencv-python-headless==4.10.0.84", "Pillow==10.4.0", "tqdm==4.66.1", "safetensors==0.4.3"]
        subprocess.run([UV, "pip", "install", "--python", PYTHON, *PACKAGES], check=True)
    else:
        PYTHON = os.environ.get("M3_PYTHON", sys.executable)
    env_info = runtime.environment_info(PYTHON, "cuda" if IN_COLAB else "auto")
    DEVICE = env_info["device"]
    assert DEVICE in ("cuda", "mps"), "Use a GPU runtime for the full archived diffusion pipeline."
    display(env_info)
    if IN_COLAB and env_info["cuda_total_memory_bytes"] < 30 * 1024**3:
        print("Memory advisory: upstream recommends at least 32 GB. Attempting the unchanged pipeline on this GPU; if CUDA runs out of memory, use A100 and restart Run all.")
    dependency_probe = "import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ['torch','torchvision','numpy','diffusers','transformers','timm','lossers','huggingface_hub']},indent=2))"
    dependencies = json.loads(subprocess.check_output([PYTHON, "-B", "-c", dependency_probe], text=True))
    assert dependencies["torch"].split("+")[0] == "2.2.2" and dependencies["diffusers"] == "0.12.1"
    assert dependencies["lossers"] == "0.0.9", "Use the calibrated LPIPS dependency version."
    print(json.dumps(dependencies, indent=2))
    (WORK / "environment.json").write_text(json.dumps({"runtime":env_info,"packages":dependencies}, indent=2))
    """)
    code("""
    # Load author checkpoints/cache and pin the diffusion model. All bytes are verified.
    resources = json.loads((BUNDLE / "resources.json").read_text())
    if IN_COLAB:
        weights_archive = RESOURCE / "m3_weights.zip"
        WEIGHTS = WORK / "weights"
        if weights_archive.is_file():
            extract_checked(weights_archive, WEIGHTS)
        else:
            assert (RESOURCE/"weights"/"checkpoints").is_dir(), "Provide weights/ or m3_weights.zip in RESOURCE_FOLDER."
            # Copy six files once from Drive to local disk; avoid model I/O over Drive.
            for item in resources["weights"]:
                source = RESOURCE/"weights"/item["path"]
                target = WEIGHTS/item["path"]
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
        CHECKPOINT_DIR, LPIPS_CACHE = WEIGHTS / "checkpoints", WEIGHTS / ".cache"
        spec = resources["sd_model"]
        download_code = (
            "from huggingface_hub import snapshot_download; import json; "
            "p=snapshot_download(repo_id=" + repr(spec["repo_id"]) + ",revision=" + repr(spec["revision"]) +
            ",allow_patterns=" + repr([r["path"] for r in spec["files"]]) + "); print('MODEL_PATH='+p)"
        )
        dl = subprocess.run([PYTHON, "-B", "-c", download_code], check=True, text=True, stdout=subprocess.PIPE)
        print(dl.stdout)
        MODEL_DIR = Path([s.split("=",1)[1] for s in dl.stdout.splitlines() if s.startswith("MODEL_PATH=")][-1])
    else:
        CHECKPOINT_DIR = Path(os.environ["M3_CHECKPOINT_DIR"])
        LPIPS_CACHE = Path(os.environ["M3_LPIPS_CACHE"])
        MODEL_DIR = Path(os.environ["M3_MODEL_DIR"])
    for item in resources["sd_model"]["files"]:
        p = MODEL_DIR / item["path"]
        assert p.is_file() and p.stat().st_size == item["bytes"] and file_sha256(p) == item["sha256"], item["path"]
    RUNTIME = runtime.prepare_runtime(BUNDLE, WORK / "runtime", CHECKPOINT_DIR, LPIPS_CACHE)
    DATA = BUNDLE / "data"
    OUT = WORK / "new_execution"
    print("Verified five author checkpoints, LPIPS VGG, and", len(resources["sd_model"]["files"]), "diffusion model files.")
    print("Model revision:", resources["sd_model"]["revision"])
    print("Prepared fresh runtime. No attack has been run by the setup cells.")
    """)
    md("""
    ## 2. Input audit and source visibility
    Development/test filenames restart at `0000`, so split names and image contents matter, not the basename alone.
    We check both file hashes and decoded RGB content, plus distinct generator seeds and real-image provenance.
    Methods and parameters are fixed before this reproduction. We do not select successful examples after execution.
    """)
    code("""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    def rgb_hashes(kind, split):
        result = set()
        for p in (DATA / kind / split).glob("*.png"):
            with Image.open(p) as im:
                result.add(hashlib.sha256(im.convert("RGB").tobytes()).hexdigest())
        return result
    for kind in ("gen", "real"):
        assert not (rgb_hashes(kind, "dev") & rgb_hashes(kind, "test")), f"{kind} pixel overlap"
    manifests = {(k,s):pd.read_csv(DATA/k/s/"manifest.csv", dtype={"image_id":str})
                 for k in ("gen","real") for s in ("dev","test")}
    assert set(manifests["gen","dev"].seed).isdisjoint(set(manifests["gen","test"].seed))
    assert set(manifests["real","dev"].source).isdisjoint(set(manifests["real","test"].source))
    print("PASS: decoded-RGB, generator-seed, and real-source dev/test separation.")
    SHOW_N = min(N_IMAGES, 3)
    display(manifests["gen","test"].head(SHOW_N))
    fig, axes = plt.subplots(2, SHOW_N, figsize=(3*SHOW_N,6), squeeze=False)
    for row,kind in enumerate(("gen","real")):
        for col,p in enumerate(sorted((DATA/kind/"test").glob("*.png"))[:SHOW_N]):
            axes[row,col].imshow(Image.open(p)); axes[row,col].axis("off")
            axes[row,col].set_title(f"{kind}/test/{p.name}")
    fig.tight_layout(); plt.show()
    # The complete readable implementation is in the verified bundle, not encoded text.
    print("Core source SHA-256:", file_sha256(BUNDLE / "src" / "m2_stealth.py"))
    print("Only portability changes:", (BUNDLE/"PORTABILITY.md").read_text())
    """)
    md("""
    ## 3. Fresh end-to-end execution
    Each cell below starts a clean model process and actually computes the attack.
    Every final image is saved as PNG along with its original, intermediate stages, records, configuration, and hashes.
    The fresh output directory cannot be reused accidentally. Changing to 100 images reruns every method on the same full test set.
    """)
    for method in ("A5", "A2", "X2", "X4", "ENS", "CAPB_ENS", "EOT_ENS"):
        code(f"""
        # {method}: real model execution, not an archived log replay.
        result_{method} = runtime.run_method(RUNTIME, DATA, OUT, "{method}",
                                          python=PYTHON, device=DEVICE, model_dir=MODEL_DIR, max_images=N_IMAGES)
        print(json.dumps(result_{method}, indent=2))
        """)
    md("""
    ## 4. Independent evaluation of the newly saved PNGs
    Reload the saved files using a new process; do not reuse the attack loop's in-memory predictions.
    Evaluate all four detectors under no transform, JPEG75, and resize0.5. The real-image FPR uses all 100 real test images.
    LPIPS/SSIM/PSNR and the actual saved L∞ distortion are measured on the untransformed saved files.
    """)
    code("""
    fresh_evaluation = runtime.evaluate(RUNTIME, DATA, OUT, python=PYTHON, device=DEVICE,
                                        methods=METHODS, conditions=CONDITIONS, real_limit=None)
    print(json.dumps(fresh_evaluation, indent=2))
    fresh = report.load_course_results(OUT / "evaluation")
    print(f"NEW EXECUTION ONLY: {N_IMAGES} generated source images, seven methods; {DEVICE}.")
    display(report.course_main_table(fresh))
    display(report.quality_table(fresh).round(5))
    display(report.jpeg75_table(fresh).round(4))
    print("AUC audit:", report.audit_score_auc(fresh))
    """)
    code("""
    # Qualitative comparison from this new execution; display fixed image 0000.
    fig, axes = plt.subplots(2,4,figsize=(13,7))
    examples = [("Input",OUT/"A5"/"0000_originImage.png")] + [(m,OUT/m/"0000_adv_image.png") for m in METHODS]
    for ax,(title,path) in zip(axes.flat,examples):
        ax.imshow(Image.open(path)); ax.set_title(title); ax.axis("off")
    fig.suptitle("New execution · fixed example 0000 · saved PNGs", fontsize=14)
    fig.tight_layout(h_pad=2.5, rect=(0,0,1,0.95))
    (WORK/"figures").mkdir(exist_ok=True)
    fig.savefig(WORK/"figures"/"new_execution_examples.png",dpi=150,bbox_inches="tight")
    plt.show()
    print("Visual examples saved. Image-quality metrics do not establish human semantic equivalence.")
    """)
    md("""
    ## 5. Archived course results: 100 generated test images
    This section reads the **existing** course CSVs from the pre-architecture snapshot.
    It independently recounts the per-image rates and recomputes AUC. It does not claim to rerun 100-image inference.
    There are 23 evaluation labels (including two intermediate-stage aliases), not 23 independent datasets.
    The selected seven methods are the same ones run above; all archived evaluation records remain available for auditing.
    """)
    code("""
    historical = report.load_course_results(BUNDLE)
    info = report.provenance_summary(historical)
    print("HISTORICAL COURSE DATA, not fresh predictions:")
    print({k:info[k] for k in ("distinct_generated_source_ids","distinct_real_source_ids","evaluated_configurations","new_model_execution")})
    display(pd.DataFrame(info["files"]))
    print("Recomputed score-ranking AUC:", report.audit_score_auc(historical))
    display(report.course_main_table(historical))
    display(report.quality_table(historical).round(5))
    """)
    code("""
    display(report.paired_comparisons(historical, pairs=(("A5","A2"),("A5","X2"),("ENS","ENS_A2"))))
    display(report.jpeg75_table(historical).round(4))
    cap = historical.summary.query("condition == 'none' and run in ['ENS','CAPB_ENS']")
    display(cap[["run","detector","auc_real_vs_adv","mean_adv_margin","lpips"]].round(5))
    fpr = historical.real_predictions.assign(false_positive=lambda x:x.pred.eq(0)).groupby(["detector","condition"]).false_positive.mean().unstack()
    display(fpr[["none","jpeg95","jpeg75","jpeg50","resize075","resize05"]].style.format("{:.0%}"))
    print("400 paired decisions = 100 images × 4 correlated detector outcomes; not 400 independent images.")
    """)
    code("""
    figures = report.plot_course_results(historical)
    # Leave room for the legend and separate nearby labels without moving data points.
    asr_ax = figures["saved_png_asr"].axes[0]
    asr_ax.set_ylim(0,1.25)
    asr_ax.set_yticks(np.linspace(0,1,6))
    asr_ax.legend(title="Detector",ncol=4,loc="upper center",frameon=False)
    quality_ax = figures["quality_vs_held_out"].axes[0]
    quality_ax.set_xlim(-0.002,0.158)
    offsets = {"A5":(8,-17),"A2":(-28,3),"ENS":(-32,13),"EOT_ENS":(7,-12)}
    for annotation in quality_ax.texts:
        if annotation.get_text() in offsets:
            annotation.set_position(offsets[annotation.get_text()])
    for name, fig in figures.items():
        fig.savefig(WORK/"figures"/("historical_"+name+".png"),dpi=150,bbox_inches="tight")
        display(fig)
        plt.close(fig)
    print("Saved", len(figures), "historical figures, separately labelled from the fresh execution.")
    """)
    md("""
    ### What the course evidence supports
    - **Attribution:** A5 versus A2 changes 4 of 400 saved-file detector decisions; A5 versus X2 changes 259. The existing latent loop has a small decision-level increment under these settings, while the control image matters strongly. This does not prove that latent optimization is disconnected or never affects pixels/scores.
    - **CAPB:** ENS → CAPB_ENS reduces mean LPIPS from about 0.126 to 0.0093. E/R score AUC moves toward 0.5, but held-out S evasion falls from 95/98 to 37/98. Lower distortion and less overshoot come with a transfer trade-off; AUC near 0.5 is not proof of indistinguishability.
    - **EOT:** Under JPEG75, ENS → EOT_ENS improves evasion from 75/82 to 81/82 on E, 55/88 to 77/88 on R, 6/99 to 53/99 on D, and 9/97 to 18/97 on S. However real-image FPR is 11%, 27%, 30%, and 63% respectively. This is a measured JPEG improvement, not a claim of general robustness or a useful deployed operating point.
    - **Measurement:** Very high fixed-threshold ASR can coexist with reversed score ranking. Saved-file scoring, score distributions, real-image FPR, and distortion tell different parts of the story.

    Historical timing is from the original local runs and is not a cross-hardware benchmark. The original PGD8 record also remains in the archived CSVs:
    its configured optimizer bound is 8/255, but all 100 saved PNGs have a measured maximum difference of 9/255.
    It must not be described as a strictly compliant final-image 8/255 baseline. Later corrected research baselines are intentionally excluded.
    """)
    md("""
    ## 6. Peer engagement: PreDecodeGuard
    This is a separately agreed **white-box latent-access** exploratory track, not the peer's original prompt-only challenge.
    Their frozen spatial CNN receives FP16 latents. The agreed batch contains seven early-stopped PGD outputs and three unmodified controls,
    with shuffled filenames. The raw-latent bound is L∞ ≤ 0.1, measured on saved FP16 values cast to FP32, with zero tolerance.

    We transferred the lesson about overshoot into a threshold-crossing early stop; this is a conceptual transfer, not an exact reproduction of image-space CAPB.
    The peer handles fixed-VAE decoding and blind review. A bypass requires **a usable decoded image, human-confirmed visible physical violence, and a benign CNN prediction**.
    The model package identifies these as historical training-positive examples: they cannot establish held-out generalization, FPR, or AUROC.
    Seven perturbed outputs were retained from an earlier local eight-input execution; the final 7+3 batch is not a pre-registered seven-input experiment.

    **Review remains pending.** Individual scores, raw latents, model weights, filenames/control mapping, and private review files are excluded from this public-facing course bundle.
    This notebook does not rerun the peer attack or create another submission batch.
    [Agreed peer discussion](https://edstem.org/au/courses/37864/discussion/3533574?comment=7924177)
    """)
    code("""
    display(report.peer_summary_table(BUNDLE))
    peer_status = json.loads((BUNDLE/"peer"/"summary.json").read_text())
    assert peer_status["review_complete"] is False and peer_status["confirmed_count"] is None
    print("Pending review is unknown, not zero confirmed success and not 7/7 confirmed bypasses.")
    print("Only aggregate peer status is included; individual scores and the blind mapping are absent.")
    """)
    md("""
    ## 7. Individual strategies and remaining course work
    Individual ownership must reflect work actually performed. The present notebook does not retrospectively assign half of the implementation to each member.

    | Member | Current contribution status | Evidence still to add |
    |---|---|---|
    | WANG YUNJIE | EOT is the proposed technical focus; the course experiments above are available for explanation and verification | Confirm actual personal work and the distinct peer-facing strategy/evidence |
    | LIU JUNPENG | Developing an additional individual peer strategy, as agreed on 18 October | Peer target, method, implementation/test evidence, result and critical analysis |

    The rubric asks each member to discuss **one distinct strategy involving other teams**. Internal CAP/EOT experiments alone are not automatically two completed peer engagements.
    The new strategy can be added as a separate, honestly labelled section once performed. It does not require reopening later paper experiments.

    Remaining items: obtain the PreDecodeGuard human review, document actual individual contributions/peer evidence, and complete the presentation.
    Generalization to more external detectors, final-budget-matched methods, and redesigned LAO experiments remain future work outside this course artifact.
    """)
    md("""
    ## 8. Preserve the run
    The export includes newly saved images, configs, step histories, logs, CSVs, figures, environment details, and SHA-256 hashes.
    Model weights are excluded from the output archive. In Colab the archive is copied to the configured Drive folder automatically.
    **Save the notebook with outputs as well**; an output ZIP does not replace the rubric's pre-rendered cell outputs.
    """)
    code("""
    execution_status = {}
    for method in METHODS:
        record = json.loads((OUT/method/"execution.json").read_text())
        assert record["status"] == "complete" and record["n_images"] == N_IMAGES
        execution_status[method] = {"status":record["status"],"n_images":record["n_images"],"wall_seconds":record["wall_seconds"]}
    assert json.loads((OUT/"evaluation"/"execution.json").read_text())["status"] == "complete"
    final_record = {"run_id":RUN_ID,"execution_location":"Colab CUDA" if IN_COLAB else "Local Jupyter",
                    "device":DEVICE,"environment":env_info,"methods":execution_status,
                    "source_snapshot":bundle_check["source_snapshot"],"course_bundle_sha256":EXPECTED_BUNDLE_SHA256,
                    "fresh_generated_inputs":N_IMAGES,"historical_generated_inputs":100,"peer_review":"pending",
                    "personal_strategy_evidence":"incomplete; see section 7"}
    (WORK/"execution_summary.json").write_text(json.dumps(final_record,indent=2))
    export_id = uuid.uuid4().hex[:6]
    export = WORK/("export_"+export_id)
    export.mkdir()
    shutil.copytree(OUT,export/"new_execution")
    shutil.copytree(WORK/"figures",export/"figures")
    for filename in ("environment.json","execution_summary.json"):
        shutil.copyfile(WORK/filename,export/filename)
    shutil.copyfile(BUNDLE/"manifest.json",export/"source_bundle_manifest.json")
    hashes = {str(p.relative_to(export)):file_sha256(p) for p in sorted(export.rglob("*")) if p.is_file()}
    (export/"SHA256SUMS.json").write_text(json.dumps(hashes,indent=2))
    output_zip = Path(shutil.make_archive(str(WORK/f"M3_outputs_{RUN_ID}_{export_id}"),"zip",export))
    destination = RESOURCE/"outputs" if IN_COLAB else RESOURCE/"executed_outputs"
    destination.mkdir(parents=True,exist_ok=True)
    saved_zip = destination/output_zip.name
    shutil.copyfile(output_zip,saved_zip)
    print("COMPLETE: seven methods, saved-PNG rescoring, historical audit, figures, and output ZIP.")
    print("Actual hardware:", DEVICE, "| peer human review: PENDING")
    print("Output ZIP:", saved_zip)
    print("Output SHA-256:", file_sha256(saved_zip))
    display(pd.DataFrame(execution_status).T)
    """)
    md("""
    ## References and provenance
    - [StealthDiffusion official source, pinned commit a156fb0](https://github.com/wyczzy/StealthDiffusion/tree/a156fb0792732e1aabc5b94a0cbecf6829e79d73). The resource bundle keeps the upstream README and records source hashes and portability edits.
    - [Attack backbone: pinned SD2.1 mirror revision](https://huggingface.co/Manojb/stable-diffusion-2-1-base/tree/0094d483a120f3f33dafbd187ea4aa60d10de75c).
    - [Input generator SD1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and [Imagenette](https://github.com/fastai/imagenette); input manifests accompany the fixed PNGs.
    - [Google Colab FAQ](https://research.google.com/colaboratory/faq.html): notebook text/code/outputs are shared with the notebook, but runtime files are not. Keep the input bundle accessible to graders and save the notebook outputs.
    - Course source/result snapshot: `before_lao_architecture_optimization_20261018_145942`. The resource manifest verifies selected archived files and documents the three portability-only constructor/device edits.
    """)
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.10"},
        "colab":{"name":"M3_Submission.ipynb","provenance":[],"include_colab_link":False},
        "accelerator":"GPU",
        "m3_scope":{"snapshot":"before_lao_architecture_optimization_20261018_145942","paper_extensions":False,
                    "peer_human_review":"pending","execution_outputs":"not executed yet"},
    })
    nbf.validate(nb)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb,output)
    print(f"Created {output.name}: {len(cells)} cells ({sum(c.cell_type=='code' for c in cells)} code)")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delivery",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    create(a.delivery,a.output)
