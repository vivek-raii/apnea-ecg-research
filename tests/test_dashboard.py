"""Verify UI metrics track the artifact, including a changed future evaluation."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

class DashboardTests(unittest.TestCase):
    def test_metrics_follow_artifact(self):
        # Exercise a temporary app copy; never alter the real scientific results.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(ROOT / 'app.py', root / 'app.py')
            shutil.copytree(ROOT / 'src', root / 'src')
            (root / 'outputs').mkdir()
            shutil.copy(ROOT / 'outputs/results_report.html', root / 'outputs/results_report.html')
            metrics = json.loads((ROOT / 'outputs/metrics.json').read_text())
            for accuracy in [metrics['holdout']['accuracy'], 0.75]:
                metrics['holdout']['accuracy'] = accuracy
                (root / 'outputs/metrics.json').write_text(json.dumps(metrics))
                app = AppTest.from_file(str(root / 'app.py')).run(timeout=30)
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, f'{100 * accuracy:.1f}%')
                self.assertTrue(app.button[0].disabled)

if __name__ == '__main__':
    unittest.main()
