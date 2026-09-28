"""Meaningful checks for data alignment, group separation, and inference parity."""
import json, sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecg import Archive, FEATURES, WINDOW, extract_window, groups_from_metadata
from predict import predict_frame

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text())
    def test_decoder_against_independent_wfdb(self):
        import wfdb, tempfile
        a=Archive(self.config['dataset_zip'])
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as folder:
            for record in ['a01','a17','b02','c05']:
                Path(folder,record+'.apn').write_bytes(a.read(record+'.apn'))
                expected=wfdb.rdann(str(Path(folder,record)),'apn')
                actual=a.annotations(record)
                np.testing.assert_array_equal([s for s,c in actual],expected.sample)
                self.assertEqual(['A' if c==8 else 'N' for s,c in actual],expected.symbol)
        a.close()
    def test_known_duplicate_and_grouping(self):
        a=Archive(self.config['dataset_zip']); x,y=a.signal('c05'),a.signal('c06')
        n=min(len(x),len(y)-8000); np.testing.assert_array_equal(x[:n],y[8000:8000+n])
        g=groups_from_metadata(a.metadata());self.assertEqual(g['c05'],g['c06']);self.assertEqual(g['a02'],g['x14']);a.close()
    def test_flat_window_and_input_validation(self):
        f,reason,p=extract_window(np.zeros(WINDOW))
        self.assertIn('low_variation',reason); self.assertEqual(f['peak_count'],0)
        with self.assertRaises(ValueError): extract_window(np.zeros(500))
        with self.assertRaises(ValueError): extract_window(np.full(WINDOW,np.nan))
    def test_synthetic_heartbeat_spacing(self):
        # Known 1 Hz Gaussian pulses: checks detector timing without patient labels.
        t=np.arange(WINDOW)/100; x=sum(np.exp(-((t-s)/.025)**2) for s in range(1,60))
        f,reason,p=extract_window(x)
        self.assertAlmostEqual(f['rr_mean'],1,places=2)
        self.assertTrue(57<=len(p)<=60)
    def test_splits_and_saved_prediction_parity(self):
        split=json.loads((ROOT/'outputs/split_manifest.json').read_text())
        self.assertFalse(set(split['development_groups'])&set(split['holdout_groups']))
        self.assertFalse(set(split['development_records'])&set(split['holdout_records']))
        folds=json.loads((ROOT/'outputs/cv_group_manifest.json').read_text())
        for f in folds:self.assertFalse(set(f['train_groups'])&set(f['validation_groups']))
        df=pd.read_pickle(ROOT/'outputs/holdout_predictions.pkl')
        bundle=joblib.load(ROOT/'outputs/evaluation_model.joblib')
        restored=predict_frame(df,bundle)
        np.testing.assert_allclose(restored.score,df.score,atol=1e-12)
        self.assertNotIn('c06',df.record.unique())
        self.assertTrue(set(FEATURES).isdisjoint({'label','record','group','age','sex','ahi'}))
    def test_extraction_inference_parity(self):
        frame=pd.read_pickle(ROOT/'outputs/features.pkl'); row=frame.iloc[100]
        a=Archive(self.config['dataset_zip']);sig=a.signal(row.record);a.close()
        start=int(row.start_sample); f,_,_=extract_window(sig[start:start+WINDOW])
        np.testing.assert_allclose([f[k] for k in FEATURES],row[FEATURES].to_numpy(float),equal_nan=True)

if __name__=='__main__': unittest.main(verbosity=2)
