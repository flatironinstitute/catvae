import unittest
from skbio.util import get_data_path
from biom import load_table
import pandas as pd
import numpy as np
from catvae.dataset.biom import BiomDataset, Q2BiomDataset, TripletDataset
from catvae.dataset.biom import collate_q2_f

import numpy.testing as npt


class TestBiomDataset(unittest.TestCase):
    def setUp(self):
        self.table = load_table(get_data_path('table.biom'))
        self.metadata = pd.read_table(get_data_path('metadata.txt'),
                                      index_col=0)

    def test_biom(self):
        BiomDataset(self.table)

    def test_biom_getitem(self):
        data = BiomDataset(self.table, self.metadata, batch_category='batch')
        exp_sample = np.array([
            65., 66., 12., 94., 37., 43., 97., 69., 6., 22., 87., 43., 87.,
            5., 51., 53., 26., 54., 51., 76., 15., 92., 30., 43., 97., 98.,
            7., 43., 25., 51., 75., 39., 13., 90., 89., 48., 60., 79., 9.,
            97., 35., 47., 13., 44., 70., 94., 80., 62., 99., 73.
        ])
        batch = data[0]
        npt.assert_allclose(batch[0], exp_sample)
        npt.assert_allclose(batch[1], np.array(0))


class TestQ2BiomDataset(unittest.TestCase):
    def setUp(self):
        self.table = load_table(get_data_path('table.biom'))
        self.metadata = pd.read_table(get_data_path('metadata.txt'),
                                      index_col=0)
        np.random.seed(0)

    def test_biom_getitem(self):
        data = Q2BiomDataset(self.table)
        exp_sample = np.array([
            65., 66., 12., 94., 37., 43., 97., 69., 6., 22., 87., 43., 87.,
            5., 51., 53., 26., 54., 51., 76., 15., 92., 30., 43., 97., 98.,
            7., 43., 25., 51., 75., 39., 13., 90., 89., 48., 60., 79., 9.,
            97., 35., 47., 13., 44., 70., 94., 80., 62., 99., 73.
        ])
        ids = list(map(str, range(len(exp_sample))))
        exp_sample = dict(zip(ids, exp_sample))
        self.assertDictEqual(exp_sample, data[0][0])

    def test_q2_collate_f(self):
        data = Q2BiomDataset(self.table)
        batch = (data[0], data[1], data[2])
        features, sample_idx = collate_q2_f(batch)
        npt.assert_allclose(np.array([0, 1, 2]), sample_idx)
        exp_sample = np.array([
            65., 66., 12., 94., 37., 43., 97., 69., 6., 22., 87., 43., 87.,
            5., 51., 53., 26., 54., 51., 76., 15., 92., 30., 43., 97., 98.,
            7., 43., 25., 51., 75., 39., 13., 90., 89., 48., 60., 79., 9.,
            97., 35., 47., 13., 44., 70., 94., 80., 62., 99., 73.
        ])
        ids = list(map(str, range(len(exp_sample))))
        exp_sample = dict(zip(ids, exp_sample))
        self.assertDictEqual(exp_sample, features[0])


class TestTripleDataset(unittest.TestCase):
    def setUp(self):
        self.table = load_table(get_data_path('table.biom'))
        self.metadata = pd.read_table(get_data_path('metadata.txt'),
                                      index_col=0)

    def test_biom_getitem(self):
        np.random.seed(0)
        data = TripletDataset(self.table, self.metadata,
                              batch_category='batch',
                              class_category='treatment')
        batch = data[0]
        self.assertEqual(len(batch), 3)


if __name__ == '__main__':
    unittest.main()
