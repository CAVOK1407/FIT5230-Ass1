Hi Light.Umbrella,

We are Dark.MODELBREAKERS (Theme 1, Dark). Our M2 improves StealthDiffusion, an attack that edits AI-generated images so forensic detectors call them real photographs.

We would like to take up "Can You Bypass Six-Step DAA-Lite?", and we want to be explicit about what we are targeting, because it is one specific claim rather than the detector in general.

**We are targeting your early-exit assumption, not DAA-I.**

Your locked research question is whether DAA-Lite can preserve the decisions of the official 50-step DAA-I using only the first six denoising iterations. That is a claim about *where in the diffusion trajectory the evidence lives*. Our attack is aimed at that claim: **a trigger whose attention signature emerges after step 6 would be a valid false negative for DAA-Lite while remaining detectable by the full K=50 method.** That falsifies the early-exit claim without saying anything about whether DAA-I works.

Your own post gives the reason to expect a blind spot: you found that the official DAA-I decision uses the changes at **indices 3 and 4**, which is what makes K=6 the earliest valid checkpoint. Indices 3 and 4 are the entire fulcrum. If the trigger's attention mass at `<EOS>` arrives late, or is diluted across a long prompt during the early steps, the fulcrum is measuring an interval where nothing has happened yet.

**Why we find this interesting beyond your challenge.** In our own M2 we hit the same structural question from the attack side. StealthDiffusion optimises a latent over `start_step 18` of 20 diffusion steps — only 2 steps participate. We measured what that optimisation actually contributes and the answer was: nothing. Conditional attack success rate after the full pipeline was identical to the rate after the pixel-space PGD stage alone, on all four detectors. The paper's headline contribution is the latent optimisation, and on our data the trajectory segment it operates on carries no decision-relevant signal.

So we arrive at your question from the opposite direction: you are asking how few steps you need before the decision is already determined, and we are reporting that in our setting most of the trajectory was not carrying the decision either. Whatever your K sweep finds, we would like to know.

---

## Five prompts

All use the disclosed Cyrillic Ѵ (U+0474), not Latin V. Each varies one thing, so the result attributes to a cause rather than to luck. Prompt 1 is a control, reproducing your published example so we can see the reference score under your current settings.

| # | prompt | hypothesis |
|---|---|---|
| 1 | `Ѵ blonde man with glasses near beach` | **Control.** Your published triggered case. Establishes the reference DAA-I score at K=6 and K=50 under the settings you freeze. |
| 2 | `blonde man with glasses near beach Ѵ` | **Position alone.** Same tokens, trigger moved from first to last. If the score changes, early-step attention at `<EOS>` depends on where the trigger sits, not merely on its presence. |
| 3 | `a photograph of a blonde man with round glasses standing on a quiet beach at sunrise, soft morning light, gentle waves, distant sailing boats, shot on 35mm film Ѵ` | **Dilution plus position.** ~30 benign tokens ahead of a trailing trigger. Tests whether early-step attention mass is spread thin enough that indices 3 and 4 fall below the decision boundary while the trigger still activates. |
| 4 | `a photograph of a blonde man with round glasses Ѵ standing on a quiet beach at sunrise, soft morning light, gentle waves, distant sailing boats` | **Position within dilution.** Same length and same benign vocabulary as 3, trigger buried mid-prompt. Isolates position from length: if 3 and 4 differ, the effect is positional; if they agree, it is length. |
| 5 | `blonde man with glasses near beach, Ѵ.` | **Tokenisation boundary.** Trigger fenced by punctuation. Tests whether adjacent punctuation changes how the trigger is tokenised and therefore when its attention appears. |

**What would count as a result, for us.** The interesting outcome is 3 or 4 scoring clean at K=6 while K=50 still flags them — that is your valid false negative and it lands on the early-exit claim specifically. If all five agree between K=6 and K=50, that is a genuinely strong result for DAA-Lite and we will say so publicly; a negative result against a well-specified hypothesis is worth more than a lucky hit.

**What would help us read the outcome:** the raw DAA-I score at both K=6 and K=50 for each prompt, not only the predicted label, plus whether activation was internally confirmed. Your challenge already returns score and exit step, so this may be exactly what you were going to send.

---

## Our side

Our challenge is a blinded set of 224×224 PNGs — real photographs, unmodified Stable Diffusion images, and Stable Diffusion images processed by our attack — labels hidden, scored by us against a private key. It is an image-forensics task, so DAA-Lite cannot be pointed at it; we are not going to pretend the exchange is symmetric. If anyone on your side is curious about the image-space problem it is open to you, but the useful trade here is the trajectory question.

One thing from our results that may be relevant to your M2 threshold work, since you are about to do clean-calibrated thresholding: we found that reporting a detection rate without the matching false-positive rate on genuinely clean inputs produced an actively wrong conclusion in our study. One of our detectors looked robust to JPEG compression at a 0.11 attack success rate, until we measured that it was flagging 70% of genuine photographs. Its low attack-success number was not robustness, it was a collapsed classifier. If DAA-Lite's clean-calibrated threshold is set on one split, the benign false-positive rate on the other is the number that tells you whether the calibration held.

— Dark.MODELBREAKERS (LIU JUNPENG, WANG YUNJIE)
