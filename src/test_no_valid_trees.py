import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

from parameters import Parameters
from predict import run_predict


def _write_laz_with_pred_instances(path: Path, pred_instances: np.ndarray):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.int32))

    las = laspy.LasData(header)
    point_count = len(pred_instances)
    las.x = np.arange(point_count, dtype=np.float64)
    las.y = np.zeros(point_count, dtype=np.float64)
    las.z = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
    las.PredInstance = pred_instances
    las.write(path)


class NoValidTreesPredictionTest(unittest.TestCase):
    def _run_skip_case(self, pred_instances):
        tmp_context = tempfile.TemporaryDirectory()
        tmpdir = tmp_context.__enter__()
        self.addCleanup(tmp_context.__exit__, None, None, None)
        tmp = Path(tmpdir)
        input_path = tmp / "input.laz"
        output_dir = tmp / "out"
        _write_laz_with_pred_instances(input_path, pred_instances)

        params = Parameters(
            _cli_parse_args=[],
            dataset_path=str(input_path),
            path_las=str(input_path),
            model_path=str(tmp / "missing-model"),
            output_dir=str(output_dir),
            path_csv_lookup=str(Path(__file__).with_name("lookup.csv")),
            output_type="both",
            n_aug=1,
        )

        return output_dir, run_predict(params)

    def test_background_only_laz_writes_empty_outputs_and_laz(self):
        output_dir, result = self._run_skip_case(np.zeros(64, dtype=np.int32))
        outfile, outfile_probs, joined, data_probs_df = result

        self.assertEqual(joined.empty, True)
        self.assertEqual(data_probs_df.empty, True)
        self.assertEqual(
            list(pd.read_csv(outfile).columns),
            ["filename", "species_id", "species_prob", "tree_H", "species"],
        )

        probs_csv = pd.read_csv(outfile_probs)
        self.assertEqual(probs_csv.shape[0], 0)
        self.assertEqual(probs_csv.columns[0], "File")
        self.assertIn("Abies_alba", probs_csv.columns)
        self.assertIn("Ulmus_laevis", probs_csv.columns)

        output_laz = output_dir / "pc_with_species.laz"
        self.assertTrue(output_laz.exists())
        las = laspy.read(output_laz)
        self.assertEqual(len(las.points), 64)
        self.assertTrue(np.all(las.species_id == 255))
        self.assertTrue(np.all(las.species_prob == 0.0))

    def test_too_small_positive_instances_are_skipped(self):
        output_dir, result = self._run_skip_case(np.ones(49, dtype=np.int32))
        outfile, outfile_probs, joined, data_probs_df = result

        self.assertTrue(joined.empty)
        self.assertTrue(data_probs_df.empty)
        self.assertEqual(pd.read_csv(outfile).shape[0], 0)
        self.assertEqual(pd.read_csv(outfile_probs).shape[0], 0)
        self.assertTrue((output_dir / "pc_with_species.laz").exists())


if __name__ == "__main__":
    unittest.main()
