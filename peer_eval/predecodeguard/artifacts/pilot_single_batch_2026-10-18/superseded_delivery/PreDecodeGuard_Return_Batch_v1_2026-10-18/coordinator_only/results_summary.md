# Results for the coordinator

One fixed single-batch pilot; no additional batches or post-result retuning.

- Returned files: 10 valid; 8 attempted and 2 unchanged controls.
- Detector flips: 8/8 attempts; 8/10 total inputs.
- Steps: 1-8, mean 3.75.
- Raw-latent L-infinity: 0.0107421875-0.080078125; limit 0.1.
- All saved outputs passed the supplied official scorer. Scores match the adapter exactly.
- All 51 synthetic implementation tests passed.
- All 10 outputs still need decoding and human review. Confirmed bypass count and rate are pending.

Early stopping occurred at the first checked score below 0.6518052091598511. The fixed step size can still overshoot substantially: final attempted scores range from 0.03291889652609825 to 0.6197622418403625. This does not establish minimum perturbation or elimination of score overshoot.

This exploratory pilot uses 10 historical training examples. It does not measure held-out generalization or a false-positive rate.

Keep this summary and the score table away from content reviewers.
