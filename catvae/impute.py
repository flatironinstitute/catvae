from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def pseudocount(x):
    return x + 1


def lda_fit(x, n_components=5):
    model = LatentDirichletAllocation(
    n_components=n_components, random_state=0)
    model.fit(x)
    return model


def lda_impute(model, x):
    comps = model.components_
    # distribution of features for each topic
    feature_probs =  comps / comps.sum(axis=1)[:, np.newaxis]
    # distribution of topic proportions
    topic_probs = model.transform(x)
    # obtain mixture distribution for each sample
    probs = topic_probs @ feature_probs

    # obtain observed probabilities and unobserved probabilities
    mask = np.array((x > 0))
    masked_probs = np.ma.array(probs, mask=mask)
    obs_probs = np.array(masked_probs.sum(axis=1))
    unobs_probs = 1 - obs_probs

    # scale the imputed values by the unobserved probability
    depths = np.array(x.sum(axis=1)).squeeze()
    scale_factor = depths * obs_probs / unobs_probs
    x = np.array(x)
    imp = ((~mask).astype(np.int) * probs) * scale_factor.reshape(-1, 1)
    return x + imp
