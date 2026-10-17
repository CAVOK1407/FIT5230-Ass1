"""Build StealthDiffusion_M2_Submission.ipynb from the M2 scripts + markdown below.

Usage:  python build_m2_notebook.py --scripts scripts --out StealthDiffusion_M2_Submission.ipynb
The four scripts are embedded verbatim as %%writefile cells so the notebook is self-contained.
"""
import argparse
import json
import os
import uuid

import nbformat


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbformat.v4.new_code_cell(text.strip("\n"))


def build(scripts_dir):
    cells = []
    S = {name: open(os.path.join(scripts_dir, name)).read() for name in
         ("m2_device.py", "m2_data.py", "m2_stealth.py", "m2_evaluate.py",
          "m2_challenge.py", "m2_diag.py")}

    cells.append(md(r"""
# StealthDiffusion — Milestone 2: making the attack transfer better and survive post-processing

**FIT5230 — Theme 1 (Adversarial ML on Generative AI), Dark side (improve an existing attack)**
**Team:** `Dark.ModelBreakers` — `<Member 1>`, `<Member 2>`

**Reference paper.** Z. Zhou, K. Sun, Z. Chen, H. Kuang, X. Sun, R. Ji.
*StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Model.* ACM MM 2024. arXiv:2408.05669.
Official code: https://github.com/wyczzy/StealthDiffusion (we pin commit `a156fb0`).
Our Milestone 1 notebook (baseline reproduction): `StealthDiffusion_Baseline_Submission_1.ipynb`.

---
### What this notebook does
Milestone 1 reproduced the official attack (PGD -> latent adversarial optimisation (LAO) -> Control-VAE decode).
Milestone 2 changes the attack itself. All changes are switches of one script (`m2_stealth.py`), so every variant is run with the same data, weights, seeds and iteration count:

| Variant | Surrogate detectors in the loss | Where the LAO loss is computed | Objective | Extra terms |
|---|---|---|---|---|
| **B00** (official) | E only | ordinary VAE decode (official) | cross-entropy (official) | none |
| **B10** | E only | **final Control-VAE path** (new) | cross-entropy | none |
| **ENS** | **E + R + D ensemble** (new) | ordinary VAE decode | cross-entropy | none |
| **FULL** | E + R + D ensemble | final Control-VAE path | **margin loss** (new) | **LPIPS constraint + input diversity** (new) |

Detector **S (Swin)** is never used in any loss, so it is a true transfer target for all variants.

Sections: 3 setup (official code + weights) · 4 data · 5 attack runs · 6 evaluation (transfer, post-processing, image quality) · 7 challenge package for other teams · 8 engagement · 9 limitations.

> Runtime: Colab **A100** (40 GB). Free T4 is not enough for the diffusion back-propagation (see M1).
> Set the paths in Section 3.4 / 4.1, then *Run all*. With `N_ATTACK = 20` the four variants take roughly 40-80 minutes.
"""))

    cells.append(md(r"""
## 1. Problem statement
AI-Generated Content Stealth (AIGC-S): take an AI-generated image and change it so that a forensic detector (real vs. AI) calls it *real*, while the image still looks the same.
StealthDiffusion does this with Stable Diffusion: a small PGD step in pixel space, then optimisation of the SD latent against a surrogate detector, then decoding with a Control-VAE that removes the decoder's spectral fingerprint.

The paper leaves three things open, and these are what we attack in M2:

1. **Transfer is unstable.** With one surrogate detector, the paper's own Table 2 shows transfer ASR to UniDetection between 18.6% and 61.1% depending on the surrogate. The attack is optimised against a single model.
2. **The optimised image is not the saved image.** In the official code the LAO loss is computed on an ordinary-VAE decode of the latent (`diff_latent_attack.py`, line 435), but the image that is saved comes from the Control-VAE decoder (line 518-520). The attack can succeed on the image it optimises and still fail on the image it outputs.
3. **No post-processing evaluation.** All results in the paper are on lossless 224x224 PNG. Real images are JPEG-compressed and resized before a detector ever sees them. The paper reports PSNR/SSIM only, and LPIPS is computed in the code but never used.
"""))

    cells.append(md(r"""
## 2. What we change in the attack
Everything is inside our `m2_stealth.py`, which imports the official modules and re-implements only the LAO loop.

| Switch | Change | Why |
|---|---|---|
| `--surrogates E,R,D` | The PGD stage and the LAO loss use the **average** of several official detectors (EfficientNet-B0, ResNet-50, DeiT-B) instead of one. | Ensemble attacks transfer better in pixel space; here we move the idea into the latent space. |
| `--decode_path final` | The LAO loss is computed on the **Control-VAE decoding path** that produces the saved image (same weights, same 0.1825 scale, same PGD-derived control features). | The optimised image and the output image become the same image. |
| `--loss margin` | Objective `relu(kappa - (logit_real - logit_AI))` instead of `-CE`. | Pushes the sample *past* the decision boundary by a margin, so that a small threshold change or post-processing does not undo it. |
| `--lpips_weight 1.0` | Adds `LPIPS(output, input)` to the objective. | Keeps the perceptual change small while the attack becomes stronger. |
| `--di_prob 0.5` | Input diversity: with probability 0.5 the decoded image is randomly resized (90-100%) and padded before the surrogates see it. | Standard transfer trick (DI); also makes the perturbation less tied to one exact pixel grid. |

Not changed: DDIM inversion, 2 diffusion steps (`start_step 18` of 20), AdamW lr 1e-2, 10 LAO iterations (official code default), PGD eps 4/255 with 10 steps, Control-VAE weights, 224x224 resolution, VAE sampling seed 8888.
"""))

    cells.append(md(r"""
## 3. Setup: official code, environment, weights
Same procedure as Milestone 1: clone the official repository at a pinned commit, build a separate Python 3.10 environment (the official 2023 dependencies do not install on the current Colab Python), link the six author checkpoints from Google Drive, patch the hard-coded proxy / multi-GPU ids.
"""))

    cells.append(code(r"""
#@title 3.1  Clone the official repository (pinned commit)
from pathlib import Path
import subprocess, sys, os

REPO = Path('/content/StealthDiffusion')
COMMIT = 'a156fb0792732e1aabc5b94a0cbecf6829e79d73'
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/wyczzy/StealthDiffusion.git', str(REPO)], check=True)
    subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', COMMIT], check=True)
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == COMMIT, f'unexpected commit {actual}; use a fresh runtime'
os.chdir(REPO)
print('repository at', REPO, 'commit', actual[:7])
"""))

    cells.append(code(r"""
#@title 3.2  Separate Python 3.10 environment with the official dependency versions (~5-8 min)
import shutil
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
UV = shutil.which('uv')
PYTHON = '/content/sdenv/bin/python'
if not Path(PYTHON).exists():
    subprocess.run([UV, 'venv', '--python', '3.10', '/content/sdenv'], check=True)
subprocess.run([UV, 'pip', 'install', '--python', PYTHON, 'torch==2.2.2', 'torchvision==0.17.2',
                '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
packages = ['setuptools==80.9.0', 'numpy==1.26.4', 'diffusers==0.12.1', 'huggingface_hub==0.23.4',
            'transformers==4.30.2', 'accelerate==0.20.3', 'timm==0.6.12', 'efficientnet-pytorch==0.7.1',
            'torchattacks==3.5.1', 'lossers==0.0.9', 'eagerpy==0.30.0', 'natsort==8.4.0', 'tensorboardX==2.6.2.2',
            'scipy==1.11.4', 'scikit-learn==1.3.2', 'opencv-python-headless==4.10.0.84', 'Pillow==10.4.0',
            'tqdm==4.66.1', 'safetensors==0.4.3']
subprocess.run([UV, 'pip', 'install', '--python', PYTHON, *packages], check=True)
check = '''
import sys, torch, diffusers, timm, torchattacks, transformers
from diffusers import StableDiffusionPipeline
from lossers.lpips import LPIPS
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.cuda.is_available(), "no CUDA GPU visible - choose an A100 runtime"
print("env OK | python", sys.version.split()[0], "| torch", torch.__version__, "| diffusers", diffusers.__version__,
      "| timm", timm.__version__, "| GPU", torch.cuda.get_device_name(0))
'''
result = subprocess.run([PYTHON, '-u', '-c', check], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout[-3000:])
assert result.returncode == 0, 'environment check failed - fix the last error above before continuing'
"""))

    cells.append(code(r"""
#@title 3.3  Get the six author checkpoints (link download OR Drive mount)
# EASIEST (no Drive mount, no account juggling): in Drive, right-click the `checkpoints`
# folder -> Share -> Anyone with the link (Viewer) -> Copy link, and paste it into
# CKPT_GDRIVE_LINK below. Leave it empty to use the Drive-mount path (CKPT_SRC) instead.
CKPT_GDRIVE_LINK = ""
CKPT_SRC = "/content/drive/MyDrive/FIT5230_Ass1/StealthDiffusion/checkpoints"
DRIVE_ROOT = "/content/drive/MyDrive/FIT5230_M2_out"
import shutil, glob
os.makedirs('checkpoints', exist_ok=True)   # under /content/StealthDiffusion
need = ['efficientnet-b0.pth', 'resnet50.pth', 'deit.pth', 'swin-t.pth', 'Controlvae.pt', 'dncnn_color_blind.pth']

def link_link_files(src_dir):
    for f in need:
        s, d = os.path.join(src_dir, f), os.path.join('checkpoints', f)
        if os.path.exists(s) and not os.path.exists(d):
            try:
                os.symlink(s, d)
            except FileExistsError:
                pass

if CKPT_GDRIVE_LINK.strip():
    # ---- mount-free path: download the shared folder with gdown ----
    import subprocess as _sp
    _sp.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'gdown'], check=False)
    import gdown
    dl = '/content/_ckpt_dl'
    shutil.rmtree(dl, ignore_errors=True)
    gdown.download_folder(CKPT_GDRIVE_LINK.strip(), output=dl, quiet=False, use_cookies=False)
    for f in need:
        hits = glob.glob(os.path.join(dl, '**', f), recursive=True)
        if hits and not os.path.exists(os.path.join('checkpoints', f)):
            shutil.copy(hits[0], os.path.join('checkpoints', f))
    # outputs cannot go to Drive if it is not mounted; keep them local
    if DRIVE_ROOT.startswith('/content/drive') and not os.path.ismount('/content/drive'):
        DRIVE_ROOT = '/content/M2_out'
        print('[3.3] Drive not mounted; outputs will go to local', DRIVE_ROOT,
              '(download them from the Files panel, or zip at the end).')
else:
    # ---- Drive-mount path ----
    from google.colab import drive
    # Mount FIRST and robustly. Never create anything under /content/drive before it is
    # mounted, or later mounts fail with "Mountpoint must not already contain files".
    if not os.path.ismount('/content/drive'):
        try:
            drive.mount('/content/drive', force_remount=True)
        except Exception as e:
            print('first mount attempt failed, clearing stale /content/drive and retrying:', e)
            if os.path.isdir('/content/drive') and not os.path.ismount('/content/drive'):
                shutil.rmtree('/content/drive', ignore_errors=True)
            drive.mount('/content/drive', force_remount=True)
    assert os.path.ismount('/content/drive'), (
        'Google Drive did not mount. Use the CKPT_GDRIVE_LINK option above, or '
        'Runtime -> Disconnect and delete runtime and re-run.')

    def find_ckpt_dir(root='/content/drive/MyDrive', need=need, max_depth=6):
        # Search the mounted My Drive for the folder that holds the checkpoints.
        if not os.path.isdir(root):
            return None
        need_set = set(need)
        for dp, dns, fns in os.walk(root):
            if dp[len(root):].strip('/').count('/') >= max_depth:
                dns[:] = []
            if len(need_set & set(fns)) >= 5:
                return dp
        return None

    if not all(os.path.exists(os.path.join(CKPT_SRC, f)) for f in need):
        print(f"[3.3] not all checkpoints at CKPT_SRC={CKPT_SRC}; searching this account's My Drive ...", flush=True)
        found = find_ckpt_dir()
        if found:
            print(f"[3.3] found checkpoints at: {found}")
            CKPT_SRC = found
        else:
            top = sorted(os.listdir('/content/drive/MyDrive'))[:40] if os.path.isdir('/content/drive/MyDrive') else []
            print("[3.3] could not find the checkpoints in this account's My Drive.")
            print("      top-level of the mounted My Drive:", top)
            print("      -> This Colab is mounting an account whose My Drive does NOT contain them.")
            print("         Easiest fix: use the CKPT_GDRIVE_LINK option at the top of this cell.")
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    link_link_files(CKPT_SRC)

os.makedirs(DRIVE_ROOT, exist_ok=True)
missing = [f for f in need if not os.path.exists(os.path.join('checkpoints', f))]
for f in need:
    print(f"{f:<26}{'READY' if f not in missing else 'MISSING':>9}")
assert not missing, ('checkpoints missing; paste the checkpoints folder share link into '
                     'CKPT_GDRIVE_LINK at the top of this cell, or fix CKPT_SRC.')
"""))

    cells.append(code(r"""
#@title 3.4  Single-GPU patch (proxy lines and cuda:1/2/3 -> cuda:0), same as M1
import re, pathlib
proxy_re = re.compile(r'^(\s*)(os\.environ\[["\']https?_proxy["\']\]\s*=)')
cvd_re   = re.compile(r'(CUDA_VISIBLE_DEVICES["\']\]\s*=\s*["\'])([1-9])(["\'])')
dev_re   = re.compile(r'cuda:([1-9])')
n = 0
for f in pathlib.Path('.').rglob('*.py'):
    if f.name.startswith('m2_'):
        continue
    out, changed = [], False
    for ln in f.read_text().splitlines(keepends=True):
        o = ln
        if proxy_re.match(ln) and not ln.lstrip().startswith('#'): ln = re.sub(proxy_re, r'\1# \2', ln)
        if cvd_re.search(ln): ln = cvd_re.sub(r'\g<1>0\g<3>', ln)
        if dev_re.search(ln): ln = dev_re.sub('cuda:0', ln)
        out.append(ln); changed = changed or (ln != o)
    if changed: f.write_text(''.join(out)); n += 1
print('patched files:', n)
"""))

    cells.append(md(r"""
### 3.5 Our M2 code
The next cells write our scripts into the repository directory. They run inside the Python 3.10 environment (`/content/sdenv/bin/python`) because they import the official modules.

- `m2_data.py` — real photos (Imagenette) + Stable Diffusion images, detector sanity check
- `m2_stealth.py` — the attack with the M2 switches (defaults = official pipeline)
- `m2_evaluate.py` — evaluation of the saved PNGs: transfer, post-processing, quality, AUC
- `m2_challenge.py` — challenge package for other teams + scorer
- `m2_diag.py` — decomposes a run stage by stage (clean -> PGD -> plain VAE -> Control-VAE)
"""))

    for name in ("m2_device.py", "m2_data.py", "m2_stealth.py", "m2_evaluate.py",
                 "m2_challenge.py", "m2_diag.py"):
        cells.append(code(f"%%writefile /content/StealthDiffusion/{name}\n" + S[name]))

    cells.append(code(r"""
#@title 3.6  Syntax check of the six scripts
for name in ['m2_device.py', 'm2_data.py', 'm2_stealth.py', 'm2_evaluate.py',
             'm2_challenge.py', 'm2_diag.py']:
    subprocess.run([PYTHON, '-m', 'py_compile', name], check=True)
    print('ok', name)
"""))

    cells.append(md(r"""
## 4. Data
The official evaluation uses GenImage (100 GB per generator). For M2 we build a small, fully documented set:

- **Real photos:** Imagenette (public 10-class ImageNet subset), validation images, centre-cropped to 224x224.
- **AI images:** Stable Diffusion, prompt `"a photo of a <class>"` for the same 10 classes, 512x512, then resized to 224x224 (the same resolution the official attack uses). GenImage's SD subsets were built the same way, so the official detectors should recognise them.
- `dev` split (10 + 10 images) is only used to check the detectors; `test` split is used for all results.

Every image has a manifest line (source file, or prompt + seed + model).
"""))

    cells.append(code(r"""
#@title 4.1  Build the dataset (real photos + generated images)
N_DEV = 10
N_TEST = 100
GEN_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_MODEL  = "Manojb/stable-diffusion-2-1-base"
REAL_SRC  = ""
# REAL_SRC: leave empty to download Imagenette automatically. If the download fails
# (or you prefer your own real photos), set it to a folder of real images on Drive,
# e.g. a GenImage ".../val/nature" folder; it is searched recursively.
DATA = Path('/content/StealthDiffusion/data')

def run_step(cmd):
    # Show the child's real output/error inline instead of a bare CalledProcessError.
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    return r.returncode

if not (DATA / 'real' / 'test' / 'manifest.csv').exists():
    cmd = [PYTHON, 'm2_data.py', 'real', '--out', str(DATA), '--n_dev', str(N_DEV), '--n_test', str(N_TEST)]
    if REAL_SRC.strip():
        cmd += ['--real_src', REAL_SRC.strip()]
    if run_step(cmd) != 0:
        raise RuntimeError('real-image step failed; read the output above. '
                           'If it is a download error, set REAL_SRC to a folder of real images on Drive and re-run.')
if not (DATA / 'gen' / 'test' / 'manifest.csv').exists():
    if run_step([PYTHON, 'm2_data.py', 'generate', '--out', str(DATA), '--n_dev', str(N_DEV), '--n_test', str(N_TEST),
                 '--model', GEN_MODEL]) != 0:
        print(f'\n!!! generation with {GEN_MODEL} failed (see output above); retrying with {SD_MODEL}. '
              'The manifest records which model produced the images.')
        if run_step([PYTHON, 'm2_data.py', 'generate', '--out', str(DATA), '--n_dev', str(N_DEV), '--n_test', str(N_TEST),
                     '--model', SD_MODEL]) != 0:
            raise RuntimeError('image generation failed with both models; read the output above.')
import pandas as pd
print(pd.read_csv(DATA / 'gen' / 'test' / 'manifest.csv').head(3).to_string())
print('real test:', len(list((DATA / 'real' / 'test').glob('*.png'))), '| generated test:', len(list((DATA / 'gen' / 'test').glob('*.png'))))
"""))

    cells.append(code(r"""
#@title 4.2  Detector sanity check on the dev split (official label convention: 0 = AI, 1 = real)
subprocess.run([PYTHON, 'm2_data.py', 'check', '--out', str(DATA)], check=True)
dev_check = pd.read_csv(DATA / 'dev_check.csv')
display(dev_check)
assert (dev_check.acc_gen >= 0.5).all() and (dev_check.acc_real >= 0.5).all(), 'a detector does not work on this data; check checkpoints / polarity before attacking'
"""))

    cells.append(code(r"""
#@title 4.3  A few samples
import matplotlib.pyplot as plt
from PIL import Image
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for j in range(5):
    axes[0, j].imshow(Image.open(DATA / 'real' / 'test' / f'{j:04d}.png')); axes[0, j].set_title(f'real {j:04d}', fontsize=9)
    axes[1, j].imshow(Image.open(DATA / 'gen' / 'test' / f'{j:04d}.png'));  axes[1, j].set_title(f'generated {j:04d}', fontsize=9)
for a in axes.ravel(): a.axis('off')
plt.suptitle('Figure 1. Test images: real photos (top) and Stable Diffusion images (bottom)'); plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 5. Attack runs
Each variant attacks the same `N_ATTACK` generated test images with the same seed. Outputs go to Drive (`DRIVE_ROOT/runs/<variant>`) so that an interrupted session can be resumed (`--resume` skips finished images).

Per image the script saves the 224x224 input (`_originImage.png`), the ordinary-VAE decode of the final latent (`_plain_vae.png`, diagnostic) and the final Control-VAE output (`_adv_image.png`, the attack result), plus `records.jsonl` with the loss history and the time per image.
"""))

    cells.append(code(r"""
#@title 5.1  Run the four variants
N_ATTACK = 20
RUNS = Path(DRIVE_ROOT) / 'runs'
VARIANTS = {
    'B00':  [],                                                     # official pipeline
    'B10':  ['--decode_path', 'final'],                              # loss on the final Control-VAE path
    'ENS':  ['--surrogates', 'E,R,D'],                               # surrogate ensemble
    'FULL': ['--surrogates', 'E,R,D', '--decode_path', 'final', '--loss', 'margin',
             '--lpips_weight', '1.0', '--di_prob', '0.5'],           # all M2 changes
}
common = ['--images_root', str(DATA / 'gen' / 'test'), '--max_images', str(N_ATTACK),
          '--pretrained_diffusion_path', SD_MODEL, '--encoder_weights', './checkpoints/Controlvae.pt',
          '--iterations', '10', '--seed', '42', '--resume']
import time
for name, extra in VARIANTS.items():
    t0 = time.time()
    print(f'=== variant {name}: m2_stealth.py {" ".join(extra)} ===', flush=True)
    subprocess.run([PYTHON, '-u', 'm2_stealth.py', '--save_dir', str(RUNS / name), *common, *extra], check=True)
    print(f'=== {name} finished in {(time.time() - t0) / 60:.1f} min ===\n', flush=True)
"""))

    cells.append(code(r"""
#@title 5.2  Quick look at the run logs (in-memory predictions; the real evaluation is Section 6)
import json
rows, secs = [], {}
for name in VARIANTS:
    with open(RUNS / name / 'records.jsonl') as f:
        recs = [json.loads(l) for l in f if l.strip()]
    secs[name] = sum(r['seconds'] for r in recs) / len(recs)
    for det in ['E', 'R', 'D', 'S']:
        rows.append({'variant': name, 'detector': det,
                     'clean->AI': sum(r['clean'][det]['pred'] == 0 for r in recs) / len(recs),
                     'final->real': sum(r['final'][det]['pred'] == 1 for r in recs) / len(recs)})
display(pd.DataFrame(rows).pivot_table(index='variant', columns='detector', values=['clean->AI', 'final->real']).round(2))
print('seconds per image:', {k: round(v, 1) for k, v in secs.items()})
"""))

    cells.append(md(r"""
## 6. Evaluation
All numbers below are computed on the **saved PNG files** by `m2_evaluate.py`, for four detectors and three post-processing conditions (`none`, `jpeg75`, `resize05` = downscale to 112 px and back).

Definitions (per variant, detector, condition):
- **cond. ASR** — among generated images the detector correctly flags as AI *before* the attack (under the same condition), the fraction whose attacked version is classified as real. This is the paper's ASR with an explicit denominator.
- **real FPR** — real test photos classified as AI. Shows how strict each detector is; the attack is only meaningful if this is low.
- **AUC (real vs attacked)** — can the detector still separate real photos from attacked images by moving its threshold? 0.5 = no.
- **PSNR / SSIM / LPIPS / Linf** — distortion between the attacked image and the 224x224 input.
- Detectors E, R, D are white-box for the ENS/FULL variants; **S is held out for every variant** (transfer). For B00/B10 only E is white-box, so R, D, S are transfer targets.
"""))

    cells.append(code(r"""
#@title 6.1  Evaluate all runs
EVAL = Path(DRIVE_ROOT) / 'eval'
subprocess.run([PYTHON, '-u', 'm2_evaluate.py', '--runs', *[f'{n}={RUNS / n}' for n in VARIANTS],
                '--real_dir', str(DATA / 'real' / 'test'), '--out', str(EVAL),
                '--detectors', 'E,R,D,S', '--conditions', 'none,jpeg75,resize05'], check=True)
summary = pd.read_csv(EVAL / 'summary.csv')
per_image = pd.read_csv(EVAL / 'per_image.csv')
summary.head()
"""))

    cells.append(code(r"""
#@title 6.2  Table 1 — conditional ASR without post-processing (white-box cells marked *)
WHITE_BOX = {'B00': ['E'], 'B10': ['E'], 'ENS': ['E', 'R', 'D'], 'FULL': ['E', 'R', 'D']}
t = summary[summary.condition == 'none'].pivot(index='run', columns='detector', values='cond_asr').reindex(index=list(VARIANTS), columns=['E', 'R', 'D', 'S'])
fmt = t.copy().astype(object)
for run in t.index:
    for det in t.columns:
        v = t.loc[run, det]
        fmt.loc[run, det] = ('%.2f' % v) + ('*' if det in WHITE_BOX[run] else '')
fmt['transfer mean'] = [('%.2f' % t.loc[run, [d for d in t.columns if d not in WHITE_BOX[run]]].mean()) for run in t.index]
print('Table 1. Conditional ASR (higher = attack better). * = detector was in the attack loss (white-box).')
display(fmt)
den = summary[summary.condition == 'none'].pivot(index='run', columns='detector', values='n_clean_detected').reindex(index=list(VARIANTS), columns=['E', 'R', 'D', 'S'])
print('denominators (images detected as AI before the attack):'); display(den)
print('real-photo false positive rate per detector:'); display(summary[(summary.condition == 'none') & (summary.run == 'B00')].set_index('detector')['real_fpr'].round(3).to_frame().T)
"""))

    cells.append(code(r"""
#@title 6.3  Table 2 — robustness to JPEG and resizing (transfer detector S and mean over R, D, S)
rob = summary.pivot(index=['run', 'condition'], columns='detector', values='cond_asr').reindex(list(VARIANTS), level=0).reindex(columns=['E', 'R', 'D', 'S']).round(2)
print('Table 2. Conditional ASR under post-processing (both clean and attacked images are processed the same way).')
display(rob)
auc = summary.pivot(index=['run', 'condition'], columns='detector', values='auc_real_vs_adv').reindex(list(VARIANTS), level=0).reindex(columns=['E', 'R', 'D', 'S']).round(2)
print('Table 3. AUC real-vs-attacked (0.5 = the detector can no longer separate them by moving its threshold).')
display(auc)
"""))

    cells.append(code(r"""
#@title 6.4  Table 4 — image quality, cost and plain-vs-final regressions
q = summary[(summary.condition == 'none') & (summary.detector == 'E')].set_index('run')[['psnr', 'ssim', 'lpips', 'linf', 'seconds']].reindex(list(VARIANTS)).round(3)
q.columns = ['PSNR (dB)', 'SSIM', 'LPIPS', 'Linf', 'sec / image']
print('Table 4. Distortion of the saved attack image w.r.t. the 224x224 input, and time per image on this GPU.')
display(q)
reg = summary[summary.condition == 'none'].pivot(index='run', columns='detector', values='plain_ok_final_fail').reindex(index=list(VARIANTS), columns=['E', 'R', 'D', 'S'])
print('Table 5. Images where the ordinary-VAE decode evaded the detector but the final Control-VAE output did not (motivation for --decode_path final).')
display(reg)
"""))

    cells.append(code(r"""
#@title 6.5  Figure 2 — ASR per detector and variant; Figure 3 — robustness
import numpy as np
dets = ['E', 'R', 'D', 'S']
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
w = 0.2
for i, run in enumerate(VARIANTS):
    vals = [summary[(summary.run == run) & (summary.condition == 'none') & (summary.detector == d)].cond_asr.iloc[0] for d in dets]
    ax[0].bar(np.arange(4) + (i - 1.5) * w, vals, w, label=run)
ax[0].set_xticks(np.arange(4)); ax[0].set_xticklabels(['E (EffNet-B0)', 'R (ResNet-50)', 'D (DeiT-B)', 'S (Swin, held out)'])
ax[0].set_ylabel('conditional ASR'); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8); ax[0].set_title('Figure 2. ASR per detector (no post-processing)')
conds = ['none', 'jpeg75', 'resize05']
for i, run in enumerate(VARIANTS):
    vals = [summary[(summary.run == run) & (summary.condition == c) & (summary.detector == 'S')].cond_asr.iloc[0] for c in conds]
    ax[1].bar(np.arange(3) + (i - 1.5) * w, vals, w, label=run)
ax[1].set_xticks(np.arange(3)); ax[1].set_xticklabels(conds); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8)
ax[1].set_title('Figure 3. Transfer ASR on S under post-processing')
plt.tight_layout(); plt.show()
"""))

    cells.append(code(r"""
#@title 6.6  Figure 4 — examples (input / ordinary-VAE decode / final output) with detector verdicts
from PIL import Image
def verdicts(run, image_id, col):
    v = per_image[(per_image.run == run) & (per_image.condition == 'none') & (per_image.image_id.astype(str).str.zfill(4) == image_id)]
    return ' '.join(f"{r.detector}:{'real' if getattr(r, col) == 1 else 'AI'}" for r in v.itertuples())
show_runs = ['B00', 'FULL']
ids = sorted(p.name[:4] for p in (RUNS / 'B00').glob('*_originImage.png'))[:3]
fig, axes = plt.subplots(len(ids) * len(show_runs), 3, figsize=(9, 3.2 * len(ids) * len(show_runs)), squeeze=False)
r = 0
for image_id in ids:
    for run in show_runs:
        for j, (suffix, col, title) in enumerate([('_originImage.png', 'clean_pred', 'input'), ('_plain_vae.png', 'plain_pred', 'ordinary VAE decode'), ('_adv_image.png', 'adv_pred', 'final output')]):
            axes[r, j].imshow(Image.open(RUNS / run / f'{image_id}{suffix}'))
            axes[r, j].set_title(f'{run} {image_id} {title}\n{verdicts(run, image_id, col)}', fontsize=8); axes[r, j].axis('off')
        r += 1
plt.suptitle('Figure 4. Examples. Verdicts of the four detectors on the saved PNGs.'); plt.tight_layout(); plt.show()
"""))

    cells.append(code(r"""
#@title 6.7  Figure 5 — optimisation curves (surrogate margin per LAO iteration)
import numpy as np
fig, ax = plt.subplots(1, len(VARIANTS), figsize=(4 * len(VARIANTS), 3.2), sharey=True)
for a, name in zip(np.atleast_1d(ax), VARIANTS):
    with open(RUNS / name / 'records.jsonl') as f:
        recs = [json.loads(l) for l in f if l.strip()]
    for r in recs[:10]:
        a.plot([np.mean(h['margins']) for h in r['history']], alpha=0.6)
    a.axhline(0, color='k', lw=0.8); a.set_title(f'{name}: mean surrogate margin (real - AI)', fontsize=9); a.set_xlabel('LAO iteration')
plt.suptitle('Figure 5. Above 0 = the surrogate(s) call the decoded image real'); plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 7. Challenge package for other teams
> **Can you detect our stealthed images?**

The package mixes real photos, unmodified Stable Diffusion images and images processed by our FULL attack (labels hidden). A team returns a CSV with one label per image; we run the scorer and post detection rate on attacked images, detection rate on raw generated images and false-positive rate on real photos. Teams that prefer a white-box comparison can send us a detector wrapper (`load_detector()` returning a `torch.nn.Module` that maps RGB `[B,3,H,W]` in `[0,1]` to fake logits `[B]`); our evaluation script then treats it as one more detector.
"""))

    cells.append(code(r"""
#@title 7.1  Build the package (public zip goes to Ed; the key stays private on Drive)
CHALLENGE = Path(DRIVE_ROOT) / 'challenge'
if not (CHALLENGE / 'challenge_public.zip').exists():
    n_per_group = min(30, N_ATTACK)
    subprocess.run([PYTHON, 'm2_challenge.py', 'build', '--real_dir', str(DATA / 'real' / 'test'), '--gen_dir', str(DATA / 'gen' / 'test'),
                    '--attack_dir', str(RUNS / 'FULL'), '--n_per_group', str(n_per_group), '--out', str(CHALLENGE), '--seed', '5230'], check=True)
print(open(CHALLENGE / 'public' / 'README.md').read())
print('public zip:', CHALLENGE / 'challenge_public.zip')
print('scorer   : python m2_challenge.py score --key', CHALLENGE / 'private' / 'challenge_key.csv', '--submission <their.csv> --team <name>')
"""))

    cells.append(md(r"""
## 8. Engagement with another team
*(fill in before posting)*

- **Target team / post:** `[Light team name, Ed link]`
- **What we target:** `[e.g. their detector's reliance on frequency evidence / their purification step]`
- **What we sent:** the challenge zip (Section 7) and the detector interface above, on `[date]`.
- **What came back / our response:** `[their prediction CSV and score, or their detector run through m2_evaluate.py as an extra detector; what we changed after seeing it]`
"""))

    cells.append(md(r"""
## 9. Limitations and next steps
- Data: 10 classes, one generator, `N_ATTACK` images. GenImage-scale numbers are out of reach in the M2 window; the paper's numbers are context, not our baseline. Our baseline is B00 run on the same images.
- Detectors E, R, D are white-box for ENS/FULL. Only S (and JPEG/resize) measures transfer. Adding a public CLIP-based detector (UnivFD) as a held-out target is the next step.
- The Control-VAE is used as released; we did not retrain it. The initial PGD epsilon does not bound the final output, so we report the measured Linf/PSNR/SSIM/LPIPS of the saved image instead of claiming a budget.
- Input diversity and the margin loss add hyper-parameters (`di_prob`, `kappa`) that were set once and not tuned on the test split.
- Next (M3): EOT with a differentiable JPEG inside the LAO loop, a CLIP feature objective for semantic detectors, and query-based refinement against the Light team's detector.
"""))

    cells.append(md(r"""
## 10. References
1. Z. Zhou et al. *StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Model.* ACM MM 2024. arXiv:2408.05669.
2. M. Zhu et al. *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images.* NeurIPS 2023.
3. A. Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
4. C. Xie et al. *Improving Transferability of Adversarial Examples with Input Diversity.* CVPR 2019.
5. N. Carlini, D. Wagner. *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P 2017 (margin loss).
6. R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018 (LPIPS).
7. Y. Liu et al. *Delving into Transferable Adversarial Examples and Black-box Attacks.* ICLR 2017 (ensemble attacks).
8. J. Howard. *Imagenette.* https://github.com/fastai/imagenette
"""))

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "gpuType": "A100"},
        "accelerator": "GPU",
    }
    for c in nb.cells:
        c.id = uuid.uuid4().hex[:8]
    return nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", default="scripts")
    ap.add_argument("--out", default="StealthDiffusion_M2_Submission.ipynb")
    a = ap.parse_args()
    nb = build(a.scripts)
    nbformat.validate(nb)
    nbformat.write(nb, a.out)
    print("wrote", a.out, "cells:", len(nb.cells), "bytes:", os.path.getsize(a.out))
