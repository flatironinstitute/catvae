import unittest
from skbio.util import get_data_path
from biom import load_table
import pandas as pd
import numpy as np
from catvae.dataset.biom import BiomDataset, BiomBatchDataset
import numpy.testing as npt


class TestBiomDataset(unittest.TestCase):
    def setUp(self):
        self.table = load_table(get_data_path('table.biom'))
        self.metadata = pd.read_table(get_data_path('metadata.txt'),
                                      index_col=0)

    def test_biom(self):
        BiomDataset(self.table)

    def test_biom_getitem(self):
        data = BiomDataset(self.table, self.metadata,  batch_category='batch')
        exp_sample = np.array([
            65., 66., 12., 94., 37., 43., 97., 69.,  6., 22., 87., 43., 87.,
            5., 51., 53., 26., 54., 51., 76., 15., 92., 30., 43., 97., 98.,
            7., 43., 25., 51., 75., 39., 13., 90., 89., 48., 60., 79.,  9.,
            97., 35., 47., 13., 44., 70., 94., 80., 62., 99., 73.
        ])
        batch = data[0]
        npt.assert_allclose(batch[0], exp_sample)
        npt.assert_allclose(batch[1], np.array(0))


class TestBiomBatchDataset(unittest.TestCase):

    def setUp(self):
        self.table = load_table(get_data_path('table.biom'))
        self.metadata = pd.read_table(
            get_data_path('metadata.txt'), dtype=str)
        self.metadata = self.metadata.set_index('sampleid')
        batch_diffs = pd.read_table(
            get_data_path('batch_diffs.txt'))
        batch_diffs['featureid'] = batch_diffs['featureid'].astype(np.str)
        self.batch_differentials = batch_diffs.set_index('featureid')

    def test_populate(self):
        data = BiomBatchDataset(self.table, self.metadata,
                                self.batch_differentials,
                                batch_category='batch',
                                format_columns=False)

        self.assertEqual(data.table.shape, (50, 12))
        self.assertEqual(data.metadata.shape, (12, 5))
        self.assertEqual(data.batch_differentials.shape, (50, 2))

    def test_getitem(self):
        data = BiomBatchDataset(self.table, self.metadata,
                                self.batch_differentials,
                                batch_category='batch',
                                format_columns=False)
        batch = data[0]
        exp_sample = np.array([
            65., 66., 12., 94., 37., 43., 97., 69.,  6., 22., 87., 43., 87.,
            5., 51., 53., 26., 54., 51., 76., 15., 92., 30., 43., 97., 98.,
            7., 43., 25., 51., 75., 39., 13., 90., 89., 48., 60., 79.,  9.,
            97., 35., 47., 13., 44., 70., 94., 80., 62., 99., 73.
        ])
        exp_batch_diff = np.array([
            59.16666667, 48.16666667, 42.66666667, 75.83333333, 54.5,
            50.66666667, 60.83333333, 51.66666667, 41.83333333, 73.33333333,
            28.33333333, 30.16666667, 81.66666667, 57.66666667, 48.,
            40.5, 23.33333333, 50.5, 58., 63.16666667,
            52.33333333, 68.83333333, 35.66666667, 48., 73.5,
            43.16666667, 54.33333333, 35.16666667, 63.5, 60.33333333,
            72., 42.16666667, 38.5, 57.5, 70.83333333,
            38.16666667, 71.83333333, 68.83333333, 50.33333333, 67.16666667,
            52., 63., 26., 25., 70.83333333,
            70.66666667, 65., 47.83333333, 69.16666667, 82.])

        npt.assert_allclose(batch[0], exp_sample)
        npt.assert_allclose(batch[1], exp_batch_diff)


if __name__ == '__main__':
    unittest.main()
