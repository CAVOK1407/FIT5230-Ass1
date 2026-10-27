"""Selection leakage and quality-priority contracts; no model loading."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'StealthDiffusion'))
import m3_lao_v3_run as run


class SelectionTests(unittest.TestCase):
    def test_finite_grid_and_budgets(self):
        cs=run.candidates()
        self.assertEqual(len(cs),16)
        self.assertEqual(len({c['name'] for c in cs}),16)
        for eot in (False,True):
            ps=[c for c in cs if c['kind']=='P' and c['eot']==eot]
            self.assertEqual({(c['alpha_pixels'],c['lpips_weight']) for c in ps},
                             {(a,w) for a in (.25,.5,1.) for w in (1.,10.)})
            self.assertTrue(all(c['max_steps']==80 and c['restarts']==3 for c in ps))

    def test_quality_guard_precedes_success_and_ties_use_fixed_order(self):
        cs=[dict(name='a'),dict(name='b'),dict(name='c')]
        ms={n:dict(eligible=8,success=6,mean_lpips=.01,mean_guard=True,detector_forwards=20) for n in 'abc'}
        ms['a'].update(success=8,mean_lpips=.02,mean_guard=False)
        self.assertEqual(run.select(cs,ms)['name'],'b')
        ms['c']['detector_forwards']=19
        self.assertEqual(run.select(cs,ms)['name'],'c')
        ms['b'].update(success=7,mean_lpips=.014)
        self.assertEqual(run.select(cs,ms)['name'],'b')

    def test_no_admissible_candidates_explicit_fallback_no_grid_growth(self):
        cs=[dict(name='a'),dict(name='b')]
        ms={n:dict(eligible=8,success=s,mean_lpips=.02,mean_guard=False,detector_forwards=20) for n,s in [('a',1),('b',2)]}
        self.assertEqual(run.select(cs,ms)['name'],'b')
        self.assertEqual(len(cs),2)

    def test_empty_eligibility_cannot_rank_as_success(self):
        with self.assertRaisesRegex(RuntimeError,'Empty development eligibility'):
            run.select([dict(name='a')],dict(a=dict(eligible=0)))


if __name__=='__main__': unittest.main()
