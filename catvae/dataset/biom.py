import os
import re
import biom
import math
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List


logger = logging.getLogger(__name__)


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
            batch_category: str = None,
    ):
        super(BiomDataset).__init__()
        self.table = table
        self.metadata = metadata
        self.batch_category = batch_category
        self.populate()

    def populate(self):
        logger.info("Preprocessing dataset")

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
            batch_cats = np.unique(self.metadata[self.batch_category].values)
            batch_cats = pd.Series(
                np.arange(len(batch_cats)), index=batch_cats)
            self.batch_indices = np.array(
                list(map(lambda x: batch_cats.loc[x],
                         self.metadata[self.batch_category].values)))

        logger.info("Finished preprocessing dataset")

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


class BiomBatchDataset(BiomDataset):
    """Loads a `.biom` file.

    Parameters
    ----------
    filename : Path
        Filepath to biom table
    metadata_file : Path
        Filepath to sample metadata
    batch_differentials : str
        Pre-trained batch differentials effects
    batch_category : str
        Column name in metadata for batch indices

    Notes
    -----
    Important, periods cannot be handled in the labels
    in the batch_category. Make sure that these are converted to
    hyphens or underscores.
    """
    def __init__(
            self,
            table: biom.Table,
            metadata: pd.DataFrame,
            batch_differentials : pd.DataFrame,
            batch_category: str,
            format_columns=True,
    ):
        super(BiomBatchDataset).__init__()
        self.table = table
        self.metadata = metadata
        self.batch_category = batch_category
        self.batch_differentials = batch_differentials
        self.format_columns = format_columns
        self.populate()

    def populate(self):
        logger.info("Preprocessing dataset")
        # Match the metadata with the table
        ids = set(self.table.ids()) & set(self.metadata.index)
        filter_f = lambda v, i, m: i in ids
        self.table = self.table.filter(filter_f, axis='sample')
        self.metadata = self.metadata.loc[self.table.ids()]
        if self.metadata.index.name is None:
            raise ValueError('`Index` must have a name either'
                             '`sampleid`, `sample-id` or #SampleID')
        self.index_name = self.metadata.index.name
        self.metadata = self.metadata.reset_index()
        # Clean up the batch indexes
        if self.format_columns:
            if (self.metadata[self.batch_category].dtypes == np.float64 or
                self.metadata[self.batch_category].dtypes == np.int64):
                # format the batch category column
                m = self.metadata[self.batch_category].astype(np.int64)
                self.metadata[self.batch_category] = m.astype(np.str)
                cols = self.batch_differentials.columns
                def regex_f(x):
                    return re.findall(r"\[([A-Za-z0-9_]+).*\]", x)[0]
                cols = list(map(regex_f, cols))
                self.batch_differentials.columns = cols

        # Retrieve batch labels
        batch_cats = np.unique(self.metadata[self.batch_category].values)
        batch_cats = pd.Series(
            np.arange(len(batch_cats)), index=batch_cats)
        self.batch_indices = np.array(
            list(map(lambda x: batch_cats.loc[x],
                     self.metadata[self.batch_category].values)))
        # Clean up batch differentials
        table_features = set(self.table.ids(axis='observation'))
        batch_features = set(self.batch_differentials.index)
        ids = table_features & batch_features
        filter_f = lambda v, i, m: i in ids
        self.table = self.table.filter(filter_f, axis='observation')
        table_obs = self.table.ids(axis='observation')
        self.batch_differentials = self.batch_differentials.loc[table_obs]
        logger.info("Finished preprocessing dataset")

    def __getitem__(self, i):
        """ Returns all of the samples for a given subject.

        Returns
        -------
        counts : np.array
            OTU counts for specified samples.
        batch_indices : np.array
            Membership ids for batch samples.
        """
        sample_idx = self.table.ids()[i]
        batch_indices = self.batch_indices[i]
        counts = self.table.data(id=sample_idx, axis='sample')
        batch_diffs = self.batch_differentials
        batch_diffs = np.array(batch_diffs.iloc[:, batch_indices].values)
        return counts, batch_diffs


def collate_single_f(batch):
    counts_list = np.vstack([b[0] for b in batch])
    counts = torch.from_numpy(counts_list).float()
    return counts


def collate_batch_f(batch):
    counts_list = np.vstack([b[0] for b in batch])
    batch_diffs = np.vstack([b[1] for b in batch])
    counts = torch.from_numpy(counts_list).float()
    batch_diffs = torch.from_numpy(batch_diffs).float()
    return counts, batch_diffs
