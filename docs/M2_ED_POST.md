# [Dark.ModelBreakers] M2 — Making StealthDiffusion transfer better, and measuring what actually survives

**Team:** Dark.ModelBreakers — `<Member 1>`, `<Member 2>`
**Theme 1 · Dark** (improve an existing adversarial ML attack on generative AI)
**Reference paper:** Zhou et al., *StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Model*, ACM MM 2024. [arXiv:2408.05669](https://arxiv.org/abs/2408.05669)
**Official code:** https://github.com/wyczzy/StealthDiffusion (we pin commit `a156fb0`)
**Our M1 post:** `<link to your M1 Ed post>`
**Our code for M2:** `<Colab / repo link>`

---

## 1. The problem

The task is AI-Generated Content Stealth (AIGC-S): take an image that was made by a generator, change it so a forensic detector calls it a real photo, while it still looks the same to a person.

StealthDiffusion does this in three stages:

1. A small PGD step in pixel space (eps = 4/255).
2. Latent Adversarial Optimisation (LAO): encode into Stable Diffusion's latent space, run a short DDIM chain, and optimise the latent against a surrogate detector.
3. Decode with a Control-VAE, trained to remove the spectral fingerprint the SD decoder leaves behind.

In M1 we reproduced this pipeline. For M2 we changed the attack, and we built an evaluation that answers questions the paper does not ask. Two of our results are negative. We report them anyway, because they are the interesting ones.

## 2. Three weaknesses we target

**(a) Transfer is unstable.** In Table 2 of the paper, transfer ASR to UniDetection is 46.4% / 61.1% / 18.6% / 27.8% depending on which surrogate was used. Same method, very different results. The attack is optimised against a single detector.

**(b) The image being optimised is not the image being saved.** In the official code the LAO loss is computed on an **ordinary VAE** decode of the latent (`diff_latent_attack.py`, inside the optimisation loop), but the image written to disk comes from the **Control-VAE** decoder, with a different scaling factor (0.18215 vs 0.1825).

**(c) No post-processing evaluation.** Every number in the paper is measured on lossless 224×224 PNG. In practice an image is JPEG-compressed and resized before any detector sees it. The official code also computes LPIPS but never puts it in the objective.

## 3. What we changed

Each change is one switch on our `m2_stealth.py`, which imports the official modules and re-implements only the LAO loop. With all switches at their defaults the script reproduces the official pipeline.

| Switch | Change | Weakness |
|---|---|---|
| `--surrogates E,R,D` | The PGD stage and the LAO loss use the **average** of several official detectors instead of one. | (a) |
| `--decode_path final` | The LAO loss is computed on the **same Control-VAE path that produces the saved image** (same decoder, same 0.1825 scaling, same control features). Gradients still reach only the latent; the Control-VAE stays frozen. | (b) |
| `--loss margin` | Objective becomes `relu(kappa − (logit_real − logit_AI))` instead of `−CE`. Pushes the sample **past** the boundary by a margin instead of infinitely far. | (c) |
| `--lpips_weight 1.0` | Adds `LPIPS(output, input)` to the objective, so image quality is constrained while the attack gets stronger. | (c) |
| `--di_prob 0.5` | Input diversity: with probability 0.5 the decoded image is randomly resized to 90–100% and zero-padded back to 224 before the surrogates see it. Differentiable. | (a) |

We set `kappa = 3.0` from measurement, not by guessing. On our data DeiT gives real photographs a mean margin of 4.90, so asking an attacked image to reach +3 is asking it to look about as credible as a real photo, rather than an arbitrary overshoot.

Unchanged, so the variants stay comparable: DDIM inversion, 2 diffusion steps (`start_step` 18 of 20), AdamW lr 1e-2, 10 LAO iterations, PGD eps 4/255 × 10 steps, the released Control-VAE weights, 224×224, and the fixed sampling seed.

## 4. The four variants

| Variant | Surrogates in the loss | LAO decoding path | Objective | Extra terms |
|---|---|---|---|---|
| **B00** (official) | E | ordinary VAE | cross-entropy | none |
| **B10** | E | **final Control-VAE** | cross-entropy | none |
| **ENS** | **E + R + D** | ordinary VAE | cross-entropy | none |
| **FULL** | E + R + D | final Control-VAE | **margin** | **LPIPS + input diversity** |

E = EfficientNet-B0, R = ResNet-50, D = DeiT-Base, S = Swin-T (all four are the paper's own detectors, loaded from the authors' checkpoints).

**S is never used in any loss**, so it is a held-out target for all four variants. For B00/B10 only E is white-box, so R, D and S all measure transfer.

## 5. How we evaluate

All numbers are computed on the **saved PNG files**, re-loaded from disk — not on tensors still in memory.

- **Conditional ASR** — counted only over generated images the detector *correctly flagged as AI before the attack*, under the same post-processing condition. We print the denominator every time. The paper's ASR does not state a denominator.
- **Real-image false-positive rate** — how often each detector calls a real photo "AI". Without this, an ASR number has no reference point. This turned out to matter more than we expected (Section 7).
- **AUC (real photos vs. attacked images)** — can the detector still separate them by moving its threshold? This turned out to be the most informative number in the whole study (Section 8).
- **Post-processing robustness** — JPEG quality 75 and 0.5× downscale-upscale. Clean and attacked images go through the **same** processing, and the denominator is recomputed for each condition.
- **Image quality** — PSNR, SSIM, LPIPS and L∞ against the same 224×224 input.
- **`plain_ok_final_fail`** — images the ordinary-VAE decode evaded but the final Control-VAE output did not. This directly measures weakness (b).
- **Time per image.** The paper reports no runtime at all.

**Data.** 20 real photographs (Imagenette validation, 10 classes) and 20 Stable Diffusion 2.1-base images, all 224×224. A separate 6-image dev split was used for sanity checks and was never used for tuning.

## 6. Main result: the ensemble works

Conditional ASR on lossless PNG. `*` = detector was in the attack loss. Denominators are the images each detector caught before the attack.

| Variant | E (n=14) | R (n=12) | D (n=19) | S (n=12, held out) | mean |
|---|---|---|---|---|---|
| B00 (official) | 1.00* | 0.67 | 0.58 | 0.83 | 0.77 |
| B10 | 1.00* | 0.67 | 0.58 | 0.83 | 0.77 |
| **ENS** | 1.00* | **1.00*** | **1.00*** | **0.92** | **0.98** |
| **FULL** | 1.00* | **1.00*** | **1.00*** | **0.92** | **0.98** |

Real-photo false-positive rate is 0.00 for all four detectors in this condition, so these ASR numbers have a clean reference point.

The surrogate ensemble takes R from 0.67 to 1.00 and D from 0.58 to 1.00. More importantly **S, which is never in any loss, goes from 0.83 to 0.92** — that is a real transfer gain on a held-out detector, which is the weakness we set out to attack.

**B10 changed nothing.** `plain_ok_final_fail` is 0–2 images out of 20 in almost every cell, and the decisions are identical to B00. Weakness (b) is real in the margins but does not flip decisions at this scale. We expected this switch to matter and it did not.

## 7. Main result: the attack does not survive post-processing

Mean conditional ASR, averaged over the detectors that still work in each condition (see below):

| Variant | lossless PNG | JPEG 75 | 0.5× resize |
|---|---|---|---|
| B00 (official) | 0.77 | 0.42 (−45%) | 0.17 (−78%) |
| B10 | 0.77 | 0.38 (−51%) | 0.17 (−78%) |
| ENS | **0.98** | 0.56 (−43%) | 0.46 (−53%) |
| FULL | **0.98** | 0.54 (−45%) | 0.46 (−53%) |

**Every number in the paper is the first column.** Saving the image once as a JPEG removes roughly half the attack. Downscaling and upscaling removes more than three quarters of the official attack.

The ensemble helps here too, and we did not design it to: under resizing the official attack retains 0.17 while ours retains 0.46, close to three times better.

**Why we exclude some detectors from the mean.** Two of them stop working as detectors under post-processing, and only the real-image false-positive rate reveals it:

| condition | detector | FPR on real photos | what it means |
|---|---|---|---|
| JPEG 75 | S | **0.70** | calls 70% of real photographs "AI" |
| 0.5× resize | E | **0.55** | calls 55% of real photographs "AI", and 20/20 generated images |

S's low ASR under JPEG (0.11) is not robustness. It is a classifier that has collapsed to "everything is fake". Reporting that as "the detector resisted our attack" would be wrong. This is why the FPR column exists.

## 8. The result we did not expect: AUC says nobody was actually fooled

AUC of real photographs vs. attacked images, on lossless PNG. 0.5 means indistinguishable.

| detector | B00 | B10 | ENS | FULL |
|---|---|---|---|---|
| E | 0.00 | 0.00 | 0.00 | 0.00 |
| R | 0.80 | 0.81 | 0.00 | 0.00 |
| D | 0.93 | 0.93 | 0.17 | 0.16 |
| S | 0.98 | 0.99 | **0.96** | **0.96** |

An AUC of 0.00 does not mean the detector is confused. It means the two groups are **perfectly separable in the reversed direction**: our attacked images score as *more real than real photographs do*. E's attacked images reach a mean margin of +153, which no real photograph in our set comes close to. A defender who knows this attack exists can catch 100% of our images by flagging anything that looks too confidently real.

S is the opposite failure. Its conditional ASR is 0.92, but its AUC stays at 0.96 — the attack flips S's decisions without disturbing its ranking at all. Recalibrating S's threshold recovers nearly everything.

So the honest reading of our own headline number: **ENS reaches 0.98 mean conditional ASR on lossless PNG, and not one of the four detectors is genuinely fooled.** Three are pushed so far that they invert, and the fourth is untouched in ranking. We have not found a paper in this area that reports this, including the one we are improving on, because accuracy-based ASR cannot show it.

The cause is the cross-entropy objective, which is unbounded — there is always more loss to gain, so the optimiser never stops pushing. This is exactly what `--loss margin` was meant to fix, and it does bound the *LAO* stage. It does not help here because the PGD stage has already pushed the margin to +97 before the LAO starts, and nothing bounds PGD. Fixing that is the first thing we will do next.

## 9. Quality and cost

| Variant | PSNR | SSIM | LPIPS | L∞ | seconds/image |
|---|---|---|---|---|---|
| B00 | 34.02 | 0.907 | 0.164 | 0.183 | 10.9 |
| B10 | 32.77 | 0.906 | 0.165 | 0.189 | 10.5 |
| ENS | 33.95 | 0.907 | 0.162 | 0.188 | 11.2 |
| FULL | 33.51 | 0.908 | **0.156** | 0.189 | 11.9 |

Measured on an Apple M5 Max via MPS. FULL has the best perceptual distance despite being the strongest attack, which is what the LPIPS term is for.

**The PGD epsilon does not bound the output.** L∞ of the saved image is about 0.188, which is 48/255 — twelve times the nominal 4/255 budget. This is not a bug: the image passes through DDIM inversion, two diffusion steps and a second decoder after PGD, and none of that is norm-constrained. We report measured distortion rather than claiming a budget, and we think papers in this area should do the same.

## 10. A bug we found, and how we caught it

Our first run with real weights produced an impossible result: adding E to the loss made the attack **worse** on E, 6/6 down to 0/6.

The diagnostic that caught it was the clean-image column. A detector's prediction on an unmodified input cannot depend on which surrogates we chose, but ours did — for E and R, while D and S were bit-identical across runs. E and R are CNNs with BatchNorm; D and S are transformers with LayerNorm and zero dropout.

Cause: our `Ensemble` wrapper was a freshly constructed `nn.Module`, so `self.training` was `True`. `torchattacks` snapshots `model.training`, forces eval for the attack, then restores the snapshot — putting every detector inside the wrapper back into training mode. With batch size 1, BatchNorm then used single-image statistics and overwrote its running statistics on every forward, corrupting every later prediction. The LAO was optimising against two broken detectors.

Fix: force the wrapper to eval, force every detector back to eval after each PGD call, and `assert not det.training` inside `predict()` so the failure is loud instead of silent. `tests/bn_mode_repro.py` reproduces the mechanism in 40 lines with no StealthDiffusion imports.

We mention this because it is the kind of bug that produces plausible-looking numbers. Our clean-prediction column now agrees to the last decimal across all four variants, and that agreement is what makes the rest of this post trustworthy.

## 11. Runnable code

`<Colab link>` — open, set the checkpoint path, Run all. It clones the official repo at the pinned commit, builds the Python 3.10 environment, writes our scripts, builds the dataset, runs the four variants, evaluates, and builds the challenge package. It also runs locally on a Mac with `--device mps`; see the README in our repo.

| File | What it does |
|---|---|
| `m2_stealth.py` | The attack, with the five switches. Defaults reproduce the official pipeline. |
| `m2_data.py` | Builds the dataset, plus a detector sanity check. |
| `m2_evaluate.py` | Everything in Section 5, from the saved PNGs. |
| `m2_diag.py` | Decomposes a run stage by stage (clean → PGD → plain VAE → Control-VAE) from the saved records. |
| `m2_challenge.py` | Builds the blinded challenge package and scores returned predictions. |
| `m2_device.py` | Device selection, so the same code runs on CUDA, Apple Silicon (MPS) or CPU. |

Reproducibility: the official code is pinned to `a156fb0`; all six author checkpoints are loaded strictly; seeds, resolutions, iteration counts and the exact command line are written to `run_config.json` in every run directory; per-image predictions at all four pipeline stages are kept in `records.jsonl` so any table here can be recomputed without re-running the attack.

## 12. Interactive challenge for other teams

> **Can you detect our stealthed images?**

We release a shuffled package of 224×224 PNGs with the labels hidden, containing three groups: real photographs, unmodified Stable Diffusion images, and Stable Diffusion images processed by our FULL attack.

Label each image `1 = generated` or `0 = real`, send the CSV back, and we run our scorer and post four numbers for your method: overall accuracy, detection rate on **raw** generated images, detection rate on **attacked** images, and false-positive rate on real photos. The gap between the second and third is what our attack is worth against you.

Two hints, because we would rather you beat us properly than trip on a technicality:

- Section 8 is a real vulnerability, not a boast. Our images are *too confidently real*. A threshold works.
- Section 7 says JPEG hurts us badly. You are allowed to pre-process.

If you would rather do a white-box comparison, send us a detector wrapper instead: a `load_detector()` returning a `torch.nn.Module` whose forward takes float RGB `[B,3,H,W]` in `[0,1]` and returns fake logits `[B]` (positive = generated). We will plug it into `m2_evaluate.py` and report it separately from our own, including under JPEG and resizing.

Challenge package: `<link>`

## 13. The team and the aspect we are targeting

`<Light team name>` — `<link to their Ed post>`

We are targeting `<the specific component>`, because `<one sentence of why>`. We sent them `<what>` on `<date>` and asked for `<what>`.

`<Fill this in after you have actually exchanged something. An invitation on its own is not an interaction.>`

## 14. Limitations

We would rather state these than have them found for us.

- **Small data.** 20 attacked images, denominators of 12–19. The paper's numbers are context, not our baseline; our baseline is B00 run on exactly the same images.
- **The detectors are weak on our generator before we attack anything.** They catch only 70% (E), 60% (R), 95% (D) and 60% (S) of unmodified SD 2.1-base images. They were trained on GenImage SD 1.4/1.5, and SD 2.1 has a different decoder fingerprint, so our images are partly out of distribution for them. Conditional ASR handles this correctly by only counting images they did catch, but part of why our attack looks strong is that these detectors are not at full strength on this generator.
- **E, R and D are white-box for ENS and FULL.** Only S and the post-processing conditions measure transfer, and S is still a sibling of the other three: same paper, same training data, same wrapper. A genuinely out-of-family detector is missing.
- **We use the released Control-VAE as is.** We did not retrain it, so the paper's Noise Prototype Loss is untouched by our work.
- **`kappa` and `di_prob` were set once and not tuned** on either split.
- **Only 2 of 20 diffusion steps are optimised** (`start_step` 18), which we kept from the official code and now suspect is why the LAO contributes so little.

## 15. What we do next (M3)

Each of these comes directly out of a result above, not from a wish list.

1. **Bound the PGD stage with a margin too.** Section 8 shows the attack is detectable because it overshoots, and that the overshoot happens in PGD, before the margin objective can help. This is the first fix.
2. **Sweep `start_step`.** Only 2 diffusion steps are currently optimised. Either the LAO starts contributing, or it is structurally unnecessary — both are worth knowing.
3. **Expectation-over-transformation with a differentiable JPEG inside the LAO loop.** Section 7 measures the gap; this closes it.
4. **Add a genuinely out-of-family detector** (a CLIP-feature detector such as UnivFD) as an untouched target.
5. **Re-run against SD 1.5**, so the detectors are in distribution and at full strength, and see whether the ensemble advantage survives.
6. **Tune `kappa` and `di_prob` on the dev split**, report on test, and scale to 100–200 images.

## References

1. Z. Zhou, K. Sun, Z. Chen, H. Kuang, X. Sun, R. Ji. *StealthDiffusion.* ACM MM 2024.
2. M. Zhu et al. *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images.* NeurIPS 2023.
3. A. Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
4. C. Xie et al. *Improving Transferability of Adversarial Examples with Input Diversity.* CVPR 2019.
5. N. Carlini, D. Wagner. *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P 2017.
6. A. Athalye et al. *Synthesizing Robust Adversarial Examples.* ICML 2018.
7. R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018.
8. U. Ojha, Y. Li, Y. J. Lee. *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* CVPR 2023.
