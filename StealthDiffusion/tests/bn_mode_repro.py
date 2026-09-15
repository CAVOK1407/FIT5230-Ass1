"""Minimal reproduction of the ensemble bug found in the first real-weights run.

Nothing here imports StealthDiffusion. It shows the mechanism on three toy detectors:
a wrapper module left in training mode makes torchattacks hand the wrapped BatchNorm
detectors back in training mode, after which their predictions on a FIXED input drift.

Expected output:
    OLD ... drift is large, left in training mode: True     <- the bug
    NEW ... drift is 0.0,  left in training mode: False     <- the fix

Run:  python tests/bn_mode_repro.py
"""
import torch
import torch.nn as nn


class ToyDetector(nn.Module):
    """Stands in for EfficientNet-B0 / ResNet-50: has BatchNorm, returns [0, logit_real]."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        h = self.bn(self.conv(x)).mean(dim=(2, 3))
        raw = self.fc(h)
        return torch.cat([torch.zeros_like(raw), raw], dim=1)


class EnsembleOld(nn.Module):
    """What the first run used: a fresh nn.Module, so self.training is True."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([m(x) for m in self.models], dim=0).mean(dim=0)


class EnsembleNew(EnsembleOld):
    """The fix: force the wrapper (and therefore its children) into eval mode."""

    def __init__(self, models):
        super().__init__(models)
        self.eval()
        self.requires_grad_(False)


def trial(ensemble_cls, label):
    torch.manual_seed(0)
    detectors = [ToyDetector().eval() for _ in range(3)]   # loaded and set to eval, as in main()
    fixed_input = torch.randn(1, 3, 16, 16)
    before = detectors[0](fixed_input)[0, 1].item()

    ensemble = ensemble_cls(detectors)

    # Exactly what torchattacks does around an attack:
    given_training = ensemble.training
    ensemble.eval()
    ensemble(fixed_input)            # the PGD forward passes
    if given_training:
        ensemble.train()             # <- restores the snapshot, and this is where it goes wrong

    for _ in range(10):              # the LAO loop's forward passes, on other images
        detectors[0](torch.randn(1, 3, 16, 16))

    after = detectors[0](fixed_input)[0, 1].item()
    print(f"  {label:<5} clean logit on a FIXED input: {before:+.4f} -> {after:+.4f}   "
          f"drift {abs(after - before):.4f}   left in training mode: {detectors[0].training}")
    return abs(after - before), detectors[0].training


if __name__ == "__main__":
    print("\nA detector's prediction on a fixed input must not depend on what was forwarded before it.\n")
    drift_old, train_old = trial(EnsembleOld, "OLD")
    drift_new, train_new = trial(EnsembleNew, "NEW")
    print()
    assert train_old and drift_old > 1e-4, "repro failed: the old wrapper did not corrupt the detector"
    assert not train_new and drift_new == 0.0, "fix failed: the new wrapper still leaks training mode"
    print("  OLD reproduces the bug, NEW is exact. Fix confirmed.\n")
