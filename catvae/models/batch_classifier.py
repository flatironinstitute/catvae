from sklearn.pipeline import Pipeline
from q2_sample_classifier.classify import predict_probabilities
import numpy as np
import pandas as pd
import biom


class Q2BatchClassifier(object):
    def __init__(self, model : Pipeline, categories : pd.Series,
                 n_workers : int = 1):
        self.n_workers = n_workers
        self.sample_estimator = model
        self._categories = categories
        self.sample_estimator.set_params(est__n_jobs=self.n_workers)

    def __call__(self, feature_data : np.array):
        # Borrowed from
        # https://github.com/qiime2/q2-sample-classifier/blob/
        # master/q2_sample_classifier/classify.py#L264
        index = np.arange(len(feature_data))
        probs = predict_probabilities(self.sample_estimator, feature_data, index)
        return probs[self._categories.index].values
