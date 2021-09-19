import biom
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from patsy import dmatrix
from numba import jit
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import euclidean


class BiomDataset(Dataset):
    """Loads a `.biom` file.

    Parameters
    ----------
    filename : Path
        Filepath to biom table
    metadata_file : Path
        Filepath to sample metadata
    batch_category : str
        Column name forr batch indices
    """
    def __init__(
            self,
            table: biom.Table,
            metadata: pd.DataFrame = None,
            batch_category: str = None):
        super(BiomDataset).__init__()
        self.table = table
        self.metadata = metadata
        self.batch_category = batch_category
        self.populate()

    def populate(self):

        if self.metadata is not None:
            # match the metadata with the table
            ids = set(self.table.ids()) & set(self.metadata.index)
            filter_f = lambda v, i, m: i in ids
            self.table = self.table.filter(filter_f, axis='sample')
            self.metadata = self.metadata.loc[self.table.ids()]
            if self.metadata.index.name is None:
                raise ValueError('`Index` must have a name either'
                                 '`sampleid`, `sample-id` or #SampleID')
            self.index_name = self.metadata.index.name
            self.metadata = self.metadata.reset_index()

        self.batch_indices = None
        if self.batch_category is not None and self.metadata is not None:
            batch_cats = self.metadata[self.batch_category].unique()
            self.batch_cats = pd.Series(
                np.arange(len(batch_cats)), index=batch_cats)
            self.batch_indices = np.array(
                list(map(lambda x: self.batch_cats.loc[x],
                         self.metadata[self.batch_category].values)))

    def __len__(self) -> int:
        return len(self.table.ids())

    def __getitem__(self, i):
        """ Returns all of the samples for a given subject

        Returns
        -------
        counts : np.array
            OTU counts for specified samples.
        batch_indices : np.array
            Membership ids for batch samples. If not specified, return None.
        """
        sample_idx = self.table.ids()[i]
        if self.batch_indices is not None:
            batch_indices = self.batch_indices[i]
        else:
            batch_indices = None
        counts = self.table.data(id=sample_idx, axis='sample')
        return counts, batch_indices

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = self.__len__()

        if worker_info is None:  # single-process data loading
            for i in range(end):
                yield self.__getitem__(i)
        else:
            worker_id = worker_info.id
            w = float(worker_info.num_workers)
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                yield self.__getitem__(i)


def _sample2dict(feature_data, feature_ids, i, sample_ids=None):
    # This is taking code from here
    # https://github.com/qiime2/q2-sample-classifier/
    # blob/master/q2_sample_classifier/utilities.py#L64
    features = np.empty(1, dtype=dict)
    row = feature_data[i]
    features = {feature_ids[ix]: d
                for ix, d in zip(row.indices, row.data)}
    return features, i


class Q2BiomDataset(BiomDataset):
    """ This is specific to q2 sample classifiers. """
    def __init__(self, table: biom.Table):
        super(Q2BiomDataset).__init__()
        self.feature_ids = table.ids('observation')
        self.feature_data = table.matrix_data.T.tocsr()

    def __getitem__(self, i):
        return _sample2dict(self.feature_data, self.feature_ids, i)


class BiomTestDataset(BiomDataset):
    """Loads a `.biom` file.

    Parameters
    ----------
    filename : Path
        Filepath to biom table
    metadata_file : Path
        Filepath to sample metadata
    batch_category : str
        Column name forr batch indices
    """
    def __init__(
            self,
            table: biom.Table,
            metadata: pd.DataFrame = None,
            batch_category: str = None,
            class_category: str = None):
        self.table = table
        self.metadata = metadata
        self.batch_category = batch_category
        super(BiomDataset).__init__()
        self.class_category = class_category
        self.populate()
        self.class_labeler = LabelEncoder().fit(self.metadata[class_category])

    def __getitem__(self, i):
        """ Returns all of the samples for a given subject

        Returns
        -------
        counts : np.array
            OTU counts for specified samples.
        batch_indices : np.array
            Membership ids for batch samples. If not specified, return None.
        class_indices : np.array
            Membership ids for classes. If not specified, return None.
        """
        sample_idx = self.table.ids()[i]
        if self.batch_indices is not None:
            batch_indices = self.batch_indices[i]
        else:
            batch_indices = None

        if self.class_category is not None:
            class_indices = self.class_labeler.transform(
                [self.metadata[self.class_category][i]])
        else:
            class_indices = None

        counts = self.table.data(id=sample_idx, axis='sample')
        return counts, batch_indices, class_indices


def _get_triplet(G, category):
    """ Picks triplets based on class assignments. """
    i = np.random.randint(len(G))
    c = G.iloc[i][category]
    idx = G[category] == c
    yesC = G.loc[idx]
    notC = G.loc[~idx]
    j = np.random.randint(len(yesC))
    while G.index[i] == yesC.index[j]:
        j = np.random.randint(len(yesC))
    k = np.random.randint(len(notC))
    return G.index[i], yesC.index[j], notC.index[k]


@jit
def _get_all_triples(s):
    X = np.zeros((s.shape[0]**3, 4))
    counter = 0
    for i in range(s.shape[0]):
        for j in range(i):
            for k in range(j):
                X[counter, 0] = i
                X[counter, 1] = j
                X[counter, 2] = k
                X[counter, 3] = (s[i] == s[j]) and (s[i] != s[k])
                counter += 1
    return X[:counter]


class TripletDataset(BiomDataset):
    """Loads a `.biom` file and generates triplets
    Parameters
    ----------
    filename : Path
        Filepath to biom table
    metadata_file : Path
        Filepath to sample metadata
    category : str
        Column name for class indices
    batch : str
        Column name for batch indices
    """
    def __init__(
            self,
            table: biom.Table,
            metadata: pd.DataFrame,
            class_category: str,
            batch_category: str):
        super(TripletDataset).__init__()
        self.table = table
        self.metadata = metadata
        self.batch_category = batch_category
        self.class_category = class_category
        self.populate()
        self.metadata = self.metadata.set_index(self.index_name)
        self.batch_dict = dict(list(
            self.metadata.groupby(self.batch_category)))

    def __len__(self) -> int:
        return len(self.table.ids())

    def __getitem__(self, i):
        """ Returns all of the samples for a given subject
        Returns
        -------
        i_counts : np.array
            OTU counts for reference sample
        j_counts : np.array
            OTU counts for positive sample
        k_counts : np.array
            OTU counts for negative sample
        """
        b = self.metadata.iloc[i][self.batch_category]
        batch_group = self.batch_dict[b]
        i, j, k = _get_triplet(batch_group, self.class_category)
        i_counts = self.table.data(i, axis='sample')
        j_counts = self.table.data(j, axis='sample')
        k_counts = self.table.data(k, axis='sample')
        return i_counts, j_counts, k_counts


class TripletTestDataset(BiomDataset):
    """Loads a `.biom` file and generates triplets

    Parameters
    ----------
    filename : Path
        Filepath to biom table
    metadata_file : Path
        Filepath to sample metadata
    category : str
        Column name for class indices
    batch : str
        Column name for batch indices

    Note
    ----
    Confounders not supported yet
    """
    def __init__(
            self,
            table: biom.Table,
            metadata: pd.DataFrame,
            class_category: str,
            confounder_formula: str):
        self.batch_category = None
        super(TripletTestDataset).__init__()
        self.table = table
        self.metadata = metadata
        self.class_category = class_category
        self.confounder_formula = confounder_formula
        self.populate()
        self.metadata = self.metadata.set_index(self.index_name)
        self.class_labeler = LabelEncoder().fit(self.metadata[class_category].values)
        cc = self.class_labeler.transform(self.metadata[class_category].values)
        self.all_triples = _get_all_triples(cc)


    def __len__(self) -> int:
        return len(self.all_triples)

    def __getitem__(self, i):
        """ Returns all of the samples for a given subject

        Returns
        -------
        i_counts : np.array
            OTU counts for reference sample
        j_counts : np.array
            OTU counts for positive sample
        k_counts : np.array
            OTU counts for negative sample
        """
        i, j, k, d = self.all_triples[i]
        i = self.metadata.index[i]
        j = self.metadata.index[j]
        k = self.metadata.index[k]
        i_counts = self.table.data(i, axis='sample')
        j_counts = self.table.data(j, axis='sample')
        k_counts = self.table.data(k, axis='sample')
        return i_counts, j_counts, k_counts, d


def collate_single_f(batch):
    counts_list = np.vstack([b[0] for b in batch])
    counts = torch.from_numpy(counts_list).float()
    return counts


def collate_batch_f(batch):
    counts_list = np.vstack([b[0] for b in batch])
    batch_ids = np.vstack([b[1] for b in batch])
    counts = torch.from_numpy(counts_list).float()
    batch_ids = torch.from_numpy(batch_ids).long()
    return counts, batch_ids.squeeze()


def collate_class_f(batch):
    counts_list = np.vstack([b[0] for b in batch])
    batch_ids = np.array([b[1] for b in batch])
    class_ids = np.array([b[2] for b in batch])
    counts = torch.from_numpy(counts_list).float()
    batch_ids = torch.from_numpy(batch_ids).long()
    class_ids = torch.from_numpy(class_ids).long()
    return counts, batch_ids.squeeze(), class_ids.squeeze()


def collate_q2_f(batch):
    features = [b[0] for b in batch]
    sample_idx = np.vstack([b[1] for b in batch])
    return features, sample_idx.squeeze()


def collate_triple_f(batch):
    i_counts_list = np.vstack([b[0] for b in batch])
    j_counts_list = np.vstack([b[1] for b in batch])
    k_counts_list = np.vstack([b[2] for b in batch])
    i_counts = torch.from_numpy(i_counts_list).float()
    j_counts = torch.from_numpy(j_counts_list).float()
    k_counts = torch.from_numpy(k_counts_list).float()
    return i_counts, j_counts, k_counts


def collate_triple_test_f(batch):
    i_counts_list = np.vstack([b[0] for b in batch])
    j_counts_list = np.vstack([b[1] for b in batch])
    k_counts_list = np.vstack([b[2] for b in batch])
    d_list = np.array([b[3] for b in batch])
    c_list = np.array([b[4] for b in batch])
    i_counts = torch.from_numpy(i_counts_list).float()
    j_counts = torch.from_numpy(j_counts_list).float()
    k_counts = torch.from_numpy(k_counts_list).float()
    d_dist = torch.from_numpy(d_list).float()
    c_dist = torch.from_numpy(c_list).float()
    return i_counts, j_counts, k_counts, d_dist, c_dist
