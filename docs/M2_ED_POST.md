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

We set `kappa = 3.0` from measurement, not by guessing. On our dev split DeiT gives real photographs a mean margin of 5.42, so asking an attacked image to reach +3 is asking it to look about as credible as a real photo, rather than an arbitrary overshoot. (It did not end up mattering — see Section 8.)

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
- **Real-image false-positive rate** — how often each detector calls a real photo "AI". Without this, an ASR number has no reference point.
- **AUC (real photos vs. attacked images)** — can the detector still separate them by moving its threshold? This turned out to be the most informative number in the whole study (Section 8).
- **Post-processing robustness** — JPEG quality 75 and 0.5× downscale-upscale. Clean and attacked images go through the **same** processing, and the denominator is recomputed for each condition.
- **Image quality** — PSNR, SSIM, LPIPS and L∞ against the same 224×224 input.
- **`plain_ok_final_fail`** — images the ordinary-VAE decode evaded but the final Control-VAE output did not. This directly measures weakness (b).
- **Time per image.** The paper reports no runtime at all.

**Data.** 100 real photographs (Imagenette validation, 10 classes) and 100 Stable Diffusion 1.5 images, all 224×224; the attack is run on the first 20 generated images of the test split. A separate 10-image dev split was used for sanity checks and was never used for tuning. Before any attack, the four detectors correctly flag 18 / 14 / 20 / 20 of those 20 images as AI — those are the denominators in Section 6.

We report a second, smaller run in Section 14 that used Stable Diffusion 2.1-base instead. It is not a duplicate: it changes one conclusion, and we say which.

## 6. Main result: the ensemble works

Conditional ASR on lossless PNG. `*` = detector was in the attack loss. Denominators are the images each detector caught before the attack, out of the 20 attacked.

| Variant | E (n=18) | R (n=14) | D (n=20) | S (n=20, held out) | mean |
|---|---|---|---|---|---|
| B00 (official) | 1.00* | 0.64 | 0.45 | 1.00 | 0.77 |
| B10 | 1.00* | **0.79** | 0.45 | 1.00 | 0.81 |
| **ENS** | 1.00* | **1.00*** | **1.00*** | 1.00 | **1.00** |
| **FULL** | 1.00* | **1.00*** | **1.00*** | 1.00 | **1.00** |

Real-photo false-positive rate is 0.00 (E, S) and 0.01 (R, D), so these ASR numbers have a clean reference point.

The surrogate ensemble takes R from 0.64 to 1.00 and D from 0.45 to 1.00. **S is already at 1.00 for every variant, including the official baseline**, so on this data S cannot show a transfer gain — it is saturated, not evidence. We say so rather than quoting the mean as if all four columns were informative. Section 8 shows what S is actually doing.

**B10 is a small but real effect, and it is visible in the diagnostic rather than the headline.** The table below counts images where the ordinary-VAE decode evaded the detector but the final Control-VAE output did not — a direct measurement of weakness (b):

| Variant | E | R | D | S |
|---|---|---|---|---|
| B00 (official) | 0 | **2** | **4** | 0 |
| B10 (`--decode_path final`) | 0 | **0** | **3** | 0 |
| ENS | 0 | 0 | 0 | 0 |
| FULL | 0 | 0 | 0 | 0 |

Matching the decode path removes both R regressions and one of the four D regressions, and R's conditional ASR rises from 0.64 to 0.79. The effect is real but small: 6 images out of 80 detector-image pairs.

We report this carefully because **we first measured this switch as having no effect at all**. On a 20-image set generated with Stable Diffusion 2.1-base, B00 and B10 produced identical decisions and `plain_ok_final_fail` was 0 everywhere. The effect only became visible once we moved to SD 1.5, where the detectors are in distribution (Section 14). A null result at one scale was not a null result.

## 7. Main result: the attack does not survive post-processing

Mean conditional ASR over all four detectors:

| Variant | lossless PNG | JPEG 75 | 0.5× resize |
|---|---|---|---|
| B00 (official) | 0.77 | 0.29 (−63%) | 0.16 (−79%) |
| B10 | 0.81 | 0.26 (−68%) | 0.19 (−77%) |
| ENS | **1.00** | 0.42 (−58%) | 0.42 (−58%) |
| FULL | **1.00** | 0.41 (−58%) | 0.45 (−55%) |

Per detector, so the averages above can be checked:

| Variant | condition | E | R | D | S |
|---|---|---|---|---|---|
| B00 | JPEG 75 | 0.93 | 0.07 | 0.05 | 0.10 |
| B00 | 0.5× resize | 0.45 | 0.20 | 0.00 | 0.00 |
| ENS | JPEG 75 | 0.87 | 0.53 | 0.15 | 0.15 |
| ENS | 0.5× resize | 0.50 | 0.65 | 0.40 | 0.15 |
| FULL | JPEG 75 | 0.93 | 0.53 | 0.15 | 0.05 |
| FULL | 0.5× resize | 0.50 | 0.65 | 0.50 | 0.15 |

**Every number in the paper is the first column.** Saving the image once as a JPEG removes roughly half the attack. Downscaling and upscaling removes more than three quarters of the official attack.

**Every number in the paper is the first column.** Saving the image once as a JPEG removes roughly 60% of the attack. Downscaling and upscaling removes 79% of the official attack.

The ensemble helps here too, and we did not design it to: under resizing the official attack retains 0.16 while ours retains 0.42–0.45, close to three times better. The two changes that were meant to help with post-processing — the margin objective and input diversity — show up only in the resize column of FULL (0.45 vs ENS 0.42), which is one image. We are not claiming that as an effect.

Note what happens to D: 0.45 → 1.00 on lossless, but 0.15 under JPEG for every variant. **Whatever the attack does to D is destroyed by compression**, and no variant we built changes that.

## 8. The result we did not expect: AUC says nobody was actually fooled

AUC of real photographs vs. attacked images, on lossless PNG. 0.5 means indistinguishable.

| detector | B00 | B10 | ENS | FULL |
|---|---|---|---|---|
| E | 0.00 | 0.00 | 0.00 | 0.00 |
| R | 0.74 | 0.73 | 0.00 | 0.00 |
| D | 0.96 | 0.96 | 0.14 | 0.14 |
| S | 0.98 | 0.98 | **0.97** | **0.97** |

An AUC of 0.00 does not mean the detector is confused. It means the two groups are **perfectly separable in the reversed direction**: our attacked images score as *more real than real photographs do*. A defender who knows this attack exists can catch 100% of our images by flagging anything that looks too confidently real.

S is the opposite failure, and it is the more embarrassing one. Its conditional ASR is **1.00 for every variant including the official baseline** — by the accuracy metric we fool it completely — yet its AUC stays at **0.97**. The attack flips every one of S's decisions without disturbing its ranking at all. Recalibrating S's threshold recovers essentially everything.

So the honest reading of our own headline number: **ENS and FULL reach 1.00 mean conditional ASR on lossless PNG, and not one of the four detectors is genuinely fooled.** Three are pushed so far that they invert (AUC 0.00–0.14), and the fourth is untouched in ranking (0.97). We have not found a paper in this area that reports this, including the one we are improving on, because accuracy-based ASR cannot show it.

The cause is the cross-entropy objective, which is unbounded — there is always more loss to gain, so the optimiser never stops pushing. This is exactly what `--loss margin` was meant to fix, and it does bound the *LAO* stage. It does not help here because the PGD stage has already pushed the margin far past the boundary before the LAO starts, and nothing bounds PGD. Fixing that is the first thing we will do next.

## 9. Quality and cost

| Variant | PSNR | SSIM | LPIPS | L∞ | seconds/image |
|---|---|---|---|---|---|
| B00 | 34.15 | 0.922 | 0.126 | 0.178 | 4.7 |
| B10 | 32.59 | 0.919 | 0.126 | 0.184 | 4.7 |
| ENS | 34.06 | 0.923 | 0.126 | 0.181 | 5.8 |
| FULL | 33.65 | 0.922 | **0.122** | 0.182 | 5.9 |

Measured on a Colab T4. The same code runs on an Apple M5 Max via MPS at 10.5–11.9 s/image. FULL has the best perceptual distance despite being the strongest attack, which is what the LPIPS term is for, though the margin is small (0.122 vs 0.126).

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

We release a shuffled package of 60 PNGs at 224×224 with the labels hidden, 20 in each of three groups: real photographs, unmodified Stable Diffusion images, and Stable Diffusion images processed by our FULL attack. Image filenames are salted hashes, so the filename tells you nothing about the group.

Label each image `1 = generated` or `0 = real`, send the CSV back, and we run our scorer and post four numbers for your method: overall accuracy, detection rate on **raw** generated images, detection rate on **attacked** images, and false-positive rate on real photos. The gap between the second and third is what our attack is worth against you.

Two hints, because we would rather you beat us properly than trip on a technicality:

- Section 8 is a real vulnerability, not a boast. Our images are *too confidently real*. A threshold works.
- Section 7 says JPEG hurts us badly. You are allowed to pre-process.

If you would rather do a white-box comparison, send us a detector wrapper instead: a `load_detector()` returning a `torch.nn.Module` whose forward takes float RGB `[B,3,H,W]` in `[0,1]` and returns fake logits `[B]` (positive = generated). We will plug it into `m2_evaluate.py` and report it separately from our own, including under JPEG and resizing.

Challenge package: `<link>`

## 13. The teams and the aspects we are targeting

No Light team in this cohort works on AI-image forensic detection, which is what our attack evades. Rather than pretend otherwise, we picked two teams whose defences are decided by a *classifier threshold*, because that is the thing our M2 turned out to be about, and we went at each of them on a different axis.

### Light.Umbrella — DAA-Lite ([their post](`<link>`)) — adversarial

**Aspect targeted: the K = 6 early-exit assumption, not DAA-I itself.**

Their locked question is whether six denoising iterations preserve the decisions of the official 50-step method. That is a claim about *where in the diffusion trajectory the evidence lives*, and they report that the DAA-I decision rests on the changes at indices 3 and 4 — so those two indices are the entire fulcrum. Our attack is aimed there: a trigger whose attention signature arrives late, or is diluted across a long prompt during the early steps, should read clean at K = 6 while the full K = 50 method still flags it. That is a valid false negative that falsifies the early-exit claim specifically.

We sent five prompts on `<date>`, using the Cyrillic trigger they disclosed, each varying exactly one factor (position, dilution, position-within-dilution, tokenisation boundary) with a control reproducing their published example. We asked for raw DAA-I scores at both K = 6 and K = 50 rather than labels.

**Why this is not an arbitrary target.** We arrived at the same structural question from the attack side. StealthDiffusion optimises `start_step 18` of 20 diffusion steps — 2 steps participate — and Section 6 shows that optimisation contributes nothing: conditional ASR after the full pipeline equals conditional ASR after the PGD stage alone, on all four detectors. They are asking how few steps are needed before the decision is fixed; we are reporting that in our setting most of the trajectory was not carrying the decision either.

### The light side, literally — PreDecodeGuard-SD3.5 ([their post](`<link>`)) — collaborative

**Aspect targeted: the conditions under which their image-space baseline is measured.**

Their comparison puts a latent-space detector against JailbreakDiffBench's Multihead detector and concludes, carefully, that it is an efficiency trade-off rather than superior accuracy (AUROC 0.905 vs 0.95, benign FPR 0.065 vs 0.016). We think that comparison gives Multihead its best possible conditions: a freshly decoded, lossless, full-resolution image, which is not what an image-space detector sees in deployment.

Section 7 of this post is our measurement of exactly that gap on four image-space detectors — and Section 8 is why it matters that the benign false-positive rate moves too. A latent-space detector never experiences JPEG or resizing, because it runs before the image exists. We proposed they re-run *only* the Multihead baseline through JPEG q75 and 0.5× resize with everything else frozen; their own four numbers do not change, and if Multihead degrades their claim strengthens from "efficiency" to "efficiency and deployment robustness".

We also gave them a caveat from our own data against their premise: our `plain_ok_final_fail` counts (Section 6) show the same latent decoded two ways receiving different detector verdicts in 6 of 80 detector-image pairs, which bounds how much a latent determines about an image that does not exist yet.

We cannot run the post-processing experiment ourselves — their repository ships manifests only and the generated images are, correctly, not distributed — so that part is a proposal rather than a result. We also submitted five benign prompts to their **utility-attack** track, each probing one low-level latent statistic, on the hypothesis that a 150k-parameter CNN fitted against 13 positive examples keys on surface statistics rather than on semantics.

### What we are not claiming

Our own challenge is one-directional. Neither team can be pointed at a set of 224×224 PNGs, so nobody has attacked us back yet. An invitation is not an interaction and a challenge nobody can run is not engagement, so we are reporting what we sent and what we asked for, and the results will follow in M3.

## 14. The generator matters, and it changed one of our conclusions

We ran the whole pipeline twice with different generators. This was not planned as an experiment — the second run happened because the first one was built on a Mac where we wanted to download only one diffusion model — but it produced the most useful methodological lesson in this project.

| | run A | run B (reported above) |
|---|---|---|
| Generator | SD 2.1-base | **SD 1.5** |
| Test images | 20 | 100 |
| Detectors' clean catch rate (E/R/D/S, of 20) | 14 / 12 / 19 / 12 | **18 / 14 / 20 / 20** |

The four detectors were trained on GenImage, whose Stable Diffusion images come from SD 1.4/1.5. SD 2.1 uses a different decoder, so in run A our generated images were partly out of distribution for the detectors, and they missed 30–40% of them before we attacked anything.

**What this changed.** In run A, `--decode_path final` (B10) produced decisions *identical* to the baseline and `plain_ok_final_fail` was 0 in every cell. We wrote it up as a null result: weakness (b) exists in the margins but does not flip decisions. In run B, with the detectors at full strength, B10 removes 3 of the 6 regressions and lifts R from 0.64 to 0.79. The switch does something; run A simply could not see it.

The lesson we take from this, and the reason it is a section rather than a footnote: **a null result measured against a weakened detector is not a null result.** If we had only run A, we would have published a wrong conclusion with a straight face.

What did *not* change between runs: the ensemble's advantage (0.77 → 1.00 either way), the post-processing collapse, and the AUC finding. Those replicate across both generators.

## 15. Limitations

We would rather state these than have them found for us.

- **Small attacked set.** 100 images in the dataset, but the attack itself is run on 20, so denominators are 14–20. Conditional ASR is reported with its denominator everywhere so the reader can weight it.
- **E, R and D are white-box for ENS and FULL.** Only S measures transfer, and S is a sibling of the other three: same paper, same training data, same wrapper. A genuinely out-of-family detector is missing.
- **S is saturated.** It sits at 1.00 conditional ASR for all four variants, so it cannot rank them. Its AUC (0.97) is the only informative thing it tells us, and what it tells us is that we did not really beat it.
- **We use the released Control-VAE as is.** We did not retrain it, so the paper's Noise Prototype Loss is untouched by our work.
- **`kappa` and `di_prob` were set once and not tuned** on either split. The margin and input-diversity switches therefore have no demonstrated effect of their own — FULL beats ENS by one image in one cell, which is not evidence.
- **Only 2 of 20 diffusion steps are optimised** (`start_step` 18), kept from the official code. We suspect this is why the LAO stage contributes so little relative to PGD.

## 16. What we do next (M3)

Each of these comes directly out of a result above, not from a wish list.

1. **Bound the PGD stage with a margin too.** Section 8 shows the attack is detectable because it overshoots, and that the overshoot happens in PGD, before the margin objective can help. Driving AUC from 0.00 towards 0.5 is a harder and more meaningful target than raising ASR, which is already at 1.00.
2. **Sweep `start_step`.** Only 2 diffusion steps are currently optimised. Either the LAO starts contributing, or it is structurally unnecessary — both are worth knowing.
3. **Expectation-over-transformation with a differentiable JPEG inside the LAO loop.** Section 7 measures the gap; this closes it. D in particular is at 1.00 lossless and 0.15 under JPEG for every variant.
4. **Add a genuinely out-of-family detector** (a CLIP-feature detector such as UnivFD) as an untouched target, since S has stopped being informative.
5. **Tune `kappa` and `di_prob` on the dev split** and report on test, so the two switches that currently show nothing get a fair test.
6. **Run the attack on all 100 images**, not 20, so the denominators support the claims.

## References

1. Z. Zhou, K. Sun, Z. Chen, H. Kuang, X. Sun, R. Ji. *StealthDiffusion.* ACM MM 2024.
2. M. Zhu et al. *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images.* NeurIPS 2023.
3. A. Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
4. C. Xie et al. *Improving Transferability of Adversarial Examples with Input Diversity.* CVPR 2019.
5. N. Carlini, D. Wagner. *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P 2017.
6. A. Athalye et al. *Synthesizing Robust Adversarial Examples.* ICML 2018.
7. R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018.
8. U. Ojha, Y. Li, Y. J. Lee. *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* CVPR 2023.
