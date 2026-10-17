# Dark.ModelBreakers — Milestone 2 challenge: can you detect our stealthed images?

`images/` contains 60 PNG images (224x224) in random order:
- real photographs,
- AI-generated images (Stable Diffusion, unmodified),
- AI-generated images processed by our modified StealthDiffusion attack.

Task: label every image as **1 = generated (AI)** or **0 = real**.
Fill `submission_template.csv` (columns `image_id,pred_label`) and send it back to us on Ed;
we run the scorer against the private key and post: overall accuracy, detection rate on raw
generated images, detection rate on attacked images, and false-positive rate on real photos.

If you want a white-box / adaptive comparison instead, share a detector wrapper with us:
`load_detector() -> torch.nn.Module` whose forward takes float RGB `[B,3,H,W]` in `[0,1]`
and returns fake logits `[B]` (positive = generated).
