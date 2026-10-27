"""Small CPU correctness tests; no training or real detector evaluation."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from m3_sfa_core import (radial_logpower, quantize_rgb, canonical_rgb, attack,
                         save_png_batch, load_png, ai_margin, image_quality,
                         conditional_asr, auc_adv_vs_real)


class TestSFACore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5230)

    def test_dynamic_spectrum_gradient(self):
        for h, w in ((64, 64), (224, 224), (48, 80)):
            x = torch.rand(2, 3, h, w, requires_grad=True)
            spectrum = radial_logpower(x)
            self.assertEqual(tuple(spectrum.shape), (2, min(h,w)//2))
            grad, = torch.autograd.grad(spectrum.square().mean(), x)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_class_direction_and_explicit_surrogates(self):
        base = torch.full((1, 3, 32, 32), .7)
        for ai_class in (0,1):
            calls = []
            def logits(x, names):
                calls.append(tuple(names))
                margin = x.mean((1,2,3)) - .5
                pair = torch.stack((margin, -margin), 1)
                if ai_class == 1:
                    pair = pair.flip(1)
                return {n:pair for n in names}
            orig = ai_margin(logits(base, ('E',))['E'], ai_class)
            calls.clear()
            result = attack(base, logits, None, mode='PGD', surrogate_names=('E','R'), ai_class=ai_class, steps=5)
            self.assertEqual(calls, [('E','R')]*5)
            after = ai_margin(logits(result.images, ('E',))['E'], ai_class)
            self.assertTrue(bool((after < orig).all()))
            self.assertEqual(len(result.trace), 5)

    def test_spec_zero_detector_queries_and_spectral_improvement(self):
        base = canonical_rgb(torch.full((1,3,32,32), .8))
        target = radial_logpower(torch.full_like(base, .6)).mean(0)
        calls=[]
        def forbidden(*args):
            calls.append(1)
            raise AssertionError('SPEC queried a detector')
        before = (radial_logpower(base) - target).square().mean()
        out = attack(base, forbidden, target, mode='SPEC', surrogate_names=(), steps=4)
        after = (radial_logpower(out.images) - target).square().mean()
        self.assertEqual(calls, [])
        self.assertLess(float(after), float(before))
        self.assertEqual(out.config['surrogate_names'], [])

    def test_lambda_zero_matches_pgd_and_scale(self):
        base=canonical_rgb(torch.rand(2,3,32,32))
        def logits(x,names):
            a=x.mean((1,2,3)); pair=torch.stack((3*a, -a),1)
            return {n:pair for n in names}
        target=radial_logpower(base).mean(0)
        pgd=attack(base,logits,target,mode='PGD',steps=3)
        sfa=attack(base,logits,target,mode='SFA',lam=0,steps=3)
        self.assertTrue(torch.equal(pgd.images,sfa.images))
        expected=float(ai_margin(logits(base,('E',))['E']).mean())
        self.assertAlmostEqual(sfa.trace[0]['mean_ai_margin'],expected,places=6)
        self.assertAlmostEqual(sfa.trace[0]['objective_before_update'],expected,places=6)
        repeat=attack(base,logits,target,mode='SFA',lam=0,steps=3)
        self.assertTrue(torch.equal(sfa.images,repeat.images))

    def test_real_disk_png_exact_budget_and_hash(self):
        base=canonical_rgb(torch.rand(3,3,24,24))
        # Includes large candidate violations and near-white/near-black pixels.
        base[0,0,0,0]=0; base[0,0,0,1]=1
        candidate=(base+torch.linspace(-.2,.2,base.numel()).reshape_as(base)).clamp(0,1)
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/f'{i}.png' for i in range(3)]
            reloaded,records=save_png_batch(candidate,base,paths)
            actual=(quantize_rgb(reloaded).to(torch.int16)-quantize_rgb(base).to(torch.int16)).abs()
            self.assertLessEqual(int(actual.max()),8)
            self.assertEqual(int(actual.max()),8)
            self.assertEqual(len(records),3)
            for path,rec in zip(paths,records):
                self.assertEqual(rec['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertTrue(rec['budget_pass'])
                self.assertEqual(load_png(path).shape,(1,3,24,24))
            with self.assertRaises(FileExistsError):
                save_png_batch(candidate,base,paths)
            self.assertEqual(image_quality(base,reloaded)[0]['linf_levels'],8)
            self.assertEqual(image_quality(base,base)[0]['psnr_db'],float('inf'))

    def test_auc_direction_and_conditional_denominator(self):
        self.assertEqual(auc_adv_vs_real([3,4],[0,1]),1)
        self.assertEqual(auc_adv_vs_real([-3,-2],[0,1]),0)
        self.assertEqual(auc_adv_vs_real([0,1],[0,1]),.5)
        self.assertEqual(auc_adv_vs_real([1],[1]),.5)
        c=conditional_asr([True,True,False],[False,True,False])
        self.assertEqual(c,{'eligible':2,'flips':1,'asr':.5})

    def test_budget_zero_and_finite_validation(self):
        x=torch.rand(1,3,16,16)
        result=attack(x,None,radial_logpower(x).mean(0),mode='SPEC',budget_levels=0,steps=2)
        self.assertTrue(torch.equal(result.images,canonical_rgb(x)))
        with self.assertRaises(ValueError):
            attack(x,None,torch.zeros(32),mode='SPEC')
        with self.assertRaises(ValueError):
            attack(x,None,None,mode='PGD')
        with self.assertRaises(ValueError):
            canonical_rgb(x*float('nan'))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
