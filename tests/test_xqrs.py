"""Independent synthetic timing checks for the experimental detector."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ecg_xqrs import extract_window, WINDOW

class DetectorTests(unittest.TestCase):
    def test_known_beats_both_polarities(self):
        t=np.arange(WINDOW)/100
        truth=np.arange(1,60)
        signal=sum(np.exp(-((t-s)/.025)**2) for s in truth)
        for polarity in [1,-1]:
            features,_,peaks=extract_window(polarity*signal)
            self.assertEqual(len(peaks),len(truth))
            self.assertLess(np.max(np.abs(peaks/100-truth)),.04)
            self.assertAlmostEqual(features['rr_mean'],1,places=2)
    def test_flat_and_invalid_inputs(self):
        features,reason,peaks=extract_window(np.zeros(WINDOW))
        self.assertIn('low_variation',reason)
        self.assertEqual(len(peaks),0)
        with self.assertRaises(ValueError): extract_window(np.full(WINDOW,np.nan))
        with self.assertRaises(ValueError): extract_window(np.zeros(WINDOW-1))

if __name__=='__main__': unittest.main(verbosity=2)
