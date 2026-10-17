# M3 follow-up messages to the three Light teams

Post each as a reply in the same Ed thread as our M2 message. Keep them short: the long version is already in the thread. Fill in the dates before posting. If a team has already replied, skip its message and answer what they said instead.

---

## 1. Light.Umbrella (DAA-Lite) — if no reply yet

Hi Light.Umbrella,

Following up on the five prompts we posted on `<date>`. Our M3 presentation is in Week 12, and the part we most want to report is your result, not ours: whether prompts 3 and 4 (the diluted ones) read clean at K = 6 while K = 50 still flags them.

If running all five is too much right now, prompts 1 (control) and 3 alone would already answer the question. Raw DAA-I scores at K = 6 and K = 50 are what we need; labels alone are fine if that is easier.

If all five agree between K = 6 and K = 50, we will say so in our presentation as a result in your favour.

— Dark.MODELBREAKERS

---

## 2. PreDecodeGuard-SD3.5 — if no reply yet

Hi Osama,

Two small asks before our Week 12 presentation.

1. Did you get a chance to run the Multihead baseline through JPEG q75 and 0.5× resize on your frozen test set? Even one of the two conditions would let us say something concrete about whether the latent-space detector's advantage survives deployment conditions. If not, no problem: we will report it as a proposal.
2. For the five benign prompts we submitted to the utility track, could we get the raw scores against your 0.6528 threshold rather than labels? The margins tell us whether the guard is keying on red regions or on scene content.

— Dark.MODELBREAKERS

---

## 3. Light.Fahhh (SafeDecoding) — if our protocol reply was never posted

Post `M2_REPLY_FAHHH_EN.md` first, then this short note under it. If the reply was posted and they have not answered, post only this note.

Hi Light.Fahhh,

We ran `<N>` of the 13 benign prompts and the 4 minimal pairs on your Gradio playground on `<date>` (fixed settings: `<seed / temperature / max_new_tokens as shown in the UI>`). Headline: `<k>` of `<N>` benign prompts were refused by SafeDecoding and answered by Base, and pair P2 (`As a nurse,` prefix) `<flipped / did not flip>` the decision.

Before we report this in Week 12 we would like to check two things with you, so the number is fair to your system:

1. Are the decoding settings in the playground the ones you use for your own numbers? If not, which should we use?
2. Do you agree with our refusal rule (refusal-prefix list, plus a separate `hedged` state)? If you count refusals differently, tell us and we will recount with your rule.

Full per-prompt table attached. Happy to share the scoring script.

— Dark.MODELBREAKERS

---

## What to record for the presentation (per team)

| | sent on | what we asked for | reply on | what came back | our reading |
|---|---|---|---|---|---|
| Umbrella | | 5 prompts, scores at K=6 and K=50 | | | |
| PreDecodeGuard | | JPEG/resize rows on Multihead; raw scores for 5 prompts | | | |
| Fahhh | | decoding settings; refusal rule | | | |

A team that never replies is still a result: say what you sent, what you expected, and what their silence means for the claim (nothing — an unanswered test is not evidence either way).
