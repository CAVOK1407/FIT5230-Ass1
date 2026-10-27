"""Device compatibility shim so the CUDA-only StealthDiffusion code runs on Apple Silicon (MPS) or CPU.

The official code and our scripts call `.cuda()` in many places. Calling `enable(device)`
once at start-up redirects every `.cuda()` (on tensors and modules, including the upstream
code we import) to the chosen device, and turns the CUDA-only bookkeeping calls
(`torch.cuda.synchronize` / `empty_cache` / `manual_seed*`) into no-ops when there is no CUDA.

Usage (already wired into m2_stealth.py / m2_data.py / m2_evaluate.py):
    import m2_device
    DEV = m2_device.enable("auto")   # "auto" -> cuda if available, else mps, else cpu
"""
import os

import torch

DEVICE = "cuda"


def resolve(name="auto"):
    if name and name != "auto":
        return name
    env = os.environ.get("M2_DEVICE", "").strip()
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def local_model(model_id, ignore_fp16=True):
    """Pre-fetch a Hugging Face model, downloading only ONE weight variant, and return a local path.

    diffusers 0.12.1 downloads every variant a repo contains (fp32 `.bin` AND `.fp16.bin`
    AND `non_ema`), which is roughly twice the bytes actually needed. We fetch with ignore
    patterns and hand `from_pretrained` a local folder, so it does not download again.
    Falls back to the plain model id if anything goes wrong.
    """
    if os.path.isdir(model_id):
        return model_id
    try:
        import glob
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_id
    # Only the diffusers pipeline sub-folders. This keeps out single-file checkpoints at the
    # repo root (e.g. v2-1_512-ema-pruned.safetensors, ~5 GB) that this code never loads.
    allow = ["model_index.json",
             "unet/*", "vae/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "feature_extractor/*"]
    # One weight format only (.bin), one precision only. .safetensors here would be a second
    # copy of the same tensors; .fp16.* a third.
    base = ["*.msgpack", "*.onnx*", "*.ckpt", "*non_ema*", "*.flax*", "*.h5"]
    ignore = base + ["*.safetensors"] + (["*.fp16.*"] if ignore_fp16 else [])

    def _fetch(ig):
        return snapshot_download(model_id, allow_patterns=allow, ignore_patterns=ig)

    def _has_unet_weights(p):
        unet = os.path.join(p, "unet")
        return bool(glob.glob(os.path.join(unet, "*.bin")) or glob.glob(os.path.join(unet, "*.safetensors")))

    try:
        path = _fetch(ignore)
        # Safety nets: fall back if this repo does not ship the format/precision we filtered to.
        if not _has_unet_weights(path):
            print("[m2_device] no .bin weights in repo; retrying with safetensors", flush=True)
            path = _fetch(base + (["*.fp16.*"] if ignore_fp16 else []))
        if not _has_unet_weights(path):
            print("[m2_device] still no weights; retrying without the fp16 filter", flush=True)
            path = _fetch(base)
        if not _has_unet_weights(path):
            print("[m2_device] could not resolve weights; letting from_pretrained handle it", flush=True)
            return model_id
    except Exception as e:  # noqa: BLE001
        print(f"[m2_device] prefetch of {model_id} failed ({e}); letting from_pretrained handle it", flush=True)
        return model_id
    print(f"[m2_device] model ready at {path}", flush=True)
    return path


def release_memory():
    """Return cached allocator blocks between images.

    On MPS the caching allocator holds blocks indefinitely. With Stable Diffusion, four
    detectors, the Control-VAE and (in the FULL variant) LPIPS-VGG16 all resident, unified
    memory fills up and macOS starts compressing and swapping pages: in one 20-image run
    the per-image time went from a flat 11 s to 117 s, 169 s, 385 s and finally 977 s, with
    a single DDIM inversion step dropping from 11 it/s to 0.02 it/s. Freeing the cache
    between images keeps the timing flat, and the timing is a number we report.

    torch.cuda.empty_cache is no-opped by enable() when there is no CUDA, so the CUDA branch
    is guarded on availability rather than on DEVICE alone.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception:  # noqa: BLE001 - never let a cache hint break a run
            pass


def enable(name="auto"):
    """Resolve and install the device shim. Returns the device string."""
    global DEVICE
    DEVICE = resolve(name)
    dev = DEVICE
    if dev == "mps":
        # let unimplemented MPS ops fall back to CPU instead of erroring
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if dev != "cuda":
        torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)      # type: ignore[assignment]
        torch.nn.Module.cuda = lambda self, *a, **k: self.to(dev)   # type: ignore[assignment]
        if not torch.cuda.is_available():
            torch.cuda.synchronize = lambda *a, **k: None           # type: ignore[assignment]
            torch.cuda.empty_cache = lambda *a, **k: None           # type: ignore[assignment]
            torch.cuda.manual_seed = lambda *a, **k: None           # type: ignore[assignment]
            torch.cuda.manual_seed_all = lambda *a, **k: None       # type: ignore[assignment]
    print(f"[m2_device] using device: {dev}", flush=True)
    return dev
