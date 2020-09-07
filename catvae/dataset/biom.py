import os
import biom
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class BiomDataset(Dataset):
    """Loads a `.biom` file.

    Parameters
    ----------
    filename : Path
        Filepath to biom table
    metadata_file : Path
        Filepath to sample metadata
    formula : str
        R-style formula for specifying sample-specific covariates.
    batch_category : str
        Column name forr batch indices
    """
    def __init__(
            self,
            table: biom.Table,
            metadata: pd.DataFrame = None,
            batch_category: str = None
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
        time : np.array
            Time points for each of the subject samples.
        batch_indices : np.array
            Membership ids for batch samples. If not specified, return None.
        labels : np.array
            Sample covariates. If not specified, return None.
        """
        sample_idx = self.table.ids()[i]
        if self.batch_indices is not None:
            batch_indices = self.batch_indices[i]
        else:
            batch_indices = None

        counts = self.table.data(id=sample_idx, axis='sample')
        return counts, batch_indices


def collate_single_f(batch):
    counts_list = np.vstack([b[0] for b in batch])
    counts = torch.from_numpy(counts_list).float()
    return counts

def collate_impute_f(batch, imputer):
    counts_list = np.vstack([b[0] for b in batch])
    smoothed_counts = imputer(counts_list)
    counts = torch.from_numpy(counts_list).float()
    smoothed_counts = torch.from_numpy(smoothed_counts).float()
    return counts, smoothed_counts
