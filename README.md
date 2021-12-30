# catvae
Categorical Variational Autoencoders

# What are Categorical Variational Autoencoders?

Variational Autoencoders (VAE) are one of the state-of-the-art methods applying neural networks to perform Bayesian inference to estimate complex high dimensional distributions, with recent techniques showing that Linear VAEs are mathematically equivalent to Principal Components Analysis.


Categorical Variational Autoencoders or Multinomial Variational Autoencoders are extentions of VAEs applied to count data.  These methods can estimate the moments of the Multinomial Logistic Normal distribution with thousands of dimensions and thousands of observations in the matter of hours.

# Getting started

## Installation

The dependencies to this package can be installed as follows
```
conda install pandas scipy biom-format gneiss pytorch pytorch-lightning -c pytorch -c conda-forge -c bioconda
pip install geotorch==0.1.0
```

The development branch of catvae can be installed via
```
pip install git+https://github.com/flatironinstitute/catvae.git
```

If one wants to use the exact software dependencies used to create these models, that can be installed via
```
conda create -n catvae -f ci/env_2021.txt
```

# Pretrained models

We offer two types of models, namely those trained on Deblurred sequences, and those mapped to reference genomes from [Web of Life] (https://biocore.github.io/wol/) (WOL).
The reference genome may offer more flexibility, since it can be interoperable between different primers and metagenomics measurements.
We used the [biom-utils](https://github.com/mortonjt/biom-util) package to map deblurred sequences to the Web of Life, but this strategy 
can also be used for sequences denoised with DADA2 or UNOISE.

## Downloading pretrained deblur models

[Pretrained Mouse VAE 128 latent dimensions](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-mouse-z128-l5-deblur.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-mouse-z128-l5-deblur.tar.gz
tar -zxvf catvae-mouse-z128-l5-deblur.tar.gz
```

[Pretrained Human VAE 128 latent dimensions](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-human-z128-l5-overdispersion-deblur.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-human-z128-l5-overdispersion-deblur.tar.gz
tar -zxvf catvae-human-z128-l5-overdispersion-deblur.tar.gz
```

## Downloading deblurred training data
[Deblurred mouse dataset](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/mouse_data.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae_models/mouse_data.tar.gz
mkdir mouse_data
tar -zxvf mouse_data.tar.gz -C mouse_data
```

[Deblurred human dataset](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/human_data.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae_models/human_data.tar.gz
mkdir human_data
tar -zxvf human_data.tar.gz -C human_data
```

## Downloading pretrained deblur models

[Pretrained Mouse VAE 64 latent dimensions](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-mouse-z64-l5-wol.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-mouse-z64-l5-wol.tar.gz
tar -zxvf catvae-mouse-z128-l5-deblur.tar.gz
```

[Pretrained Human VAE 64 latent dimensions](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/wol/catvae-human-z64-l5-wol.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-human-z64-l5-overdispersion-deblur.tar.gz
tar -zxvf catvae-human-z128-l5-overdispersion-deblur.tar.gz
```

## Downloading WOL training data

[WOL mouse dataset](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/wol/mouse_data.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae_models/wol/mouse_data.tar.gz
mkdir mouse_data
tar -zxvf mouse_data.tar.gz -C mouse_data
```

[WOL human dataset](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/wol/human_data.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae_models/wol/human_data.tar.gz
mkdir human_data
tar -zxvf human_data.tar.gz -C human_data
```


## Pre processing your data

### Preprocessing for deblurred models

All of the pretrained deblur models were trained on 100bp 16S V4 deblurred data from [Qiita](https://qiita.ucsd.edu/).  To use these models on your data, either upload your data to Qiita, or process your data using deblur.  See the [qiime2 tutorial](https://docs.qiime2.org/2021.4/tutorials/moving-pictures/#option-2-deblur) for an example of how to deblur your amplicon data.  It is assumed that the deblur sequences themselves are the observation ids, so the qiime2 approach may require relabeling the biom table observation ids (see [biom.Table.update_ids](http://biom-format.org/documentation/generated/biom.table.Table.update_ids.html))

### Preprocessing for WOL models
All of the pretrained WOL models were trained from sequences that mapped 100bp 16S V4 deblurred data from [Qiita](https://qiita.ucsd.edu/) to the [WOL](https://biocore.github.io/wol/) database. To use these models, you must map your denoised data to these databases, which can be done using the utility scripts provided [here](https://github.com/mortonjt/biom-util).


## Loading VAE models

When processing your own data, it is important to note that you can only perform inference on the microbes that have been observed by the VAE.  As a result, it is critical that your data is completely aligned with the VAE. Loading the VAE model and aligning your data against the VAE can be done as follows

```python
from catvae.util import load_model
vae_model, tree = load_model('catvae-mouse-z128-l5-deblur')
```

If you want to obtain a reduced dimension representation of your data, that can be done as follows
```python
# Load your dataset and perform dimensionality reduction
import biom
from catvae.util import extract_sample_embeddings
table = biom.load_table('mouse_data/test.biom')
sample_embeds = extract_sample_embeddings(vae_model, tree, table, return_type='tensor')
```
Here, the rows are the samples and the columns are the principal component axes.
With these representations it is possible to perform standard machine learning tasks.
See [scikit-learn](https://scikit-learn.org/stable/index.html) for some examples.


You can also sample from these embeddings, which is useful for uncertainty quantification.
Below is an example of how you would do that from a given biom input.
```python
import torch
x = torch.Tensor(table.data(id='10422.12.F.8'))
vae_model.vae.sample(x)
```

If you want to extract the VAE decoder loadings to obtain co-occurrences as done in the paper, it can be done as follows
```python
from catvae.util import extract_observation_embeddings
feature_embeds = extract_observation_embeddings(vae_model, tree)
```
With this, one can compute [squared Euclidean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean.html#scipy.spatial.distance.sqeuclidean) or
[cosine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html#scipy.spatial.distance.cosine) distances with these embeddings.  See [pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) or [DistanceMatrix.from_iterable](http://scikit-bio.org/docs/0.5.1/generated/generated/skbio.stats.distance.DistanceMatrix.from_iterable.html) for information how to compute pairwise distances.

# Documentation
The documentation for the utility functions is given below.

```python
def load_model(model_path):
    """ Loads VAE model.

    Parameters
    ----------
    model_path : str
       Path to the pretrained VAE model

    Returns
    ----------
    vae_model : MultVAE
        Pretrained Multinomial VAE
    tree : skbio.TreeNode
        The tree used to train the VAE
    """

def extract_sample_embeddings(model, tree, table, return_type='dataframe'):
    """ Extracts sample embeddings from model

    Parameters
    ----------
    vae_model : MultVAE
        Pretrained Multinomial VAE
    tree : skbio.TreeNode
        The tree used to train the VAE
    table : biom.Table
        The biom table one wishes to convert to sample embeddings
    return_type : str
        Options include 'tensor', 'array', 'dataframe' (default='tensor').
        If 'tensor' is specified, a `torch.Tensor` object is returned.
        If 'array' is specified, a `numpy.array` object is returned.
        If 'dataframe' is specified, a `pandas.DataFrame` object is returned.
    """


def extract_observation_embeddings(model, tree, return_type='dataframe'):
    """ Extracts observation embeddings from model (i.e. OTUs).

    The observation embeddings are all represented in CLR coordinates.

    Parameters
    ----------
    vae_model : MultVAE
        Pretrained Multinomial VAE
    tree : skbio.TreeNode
        The tree used to train the VAE
    return_type : str
        Options include 'tensor', 'array', 'dataframe' (default='dataframe')
    """
```
## Training the VAE models

Please refer to the Jupyter notebooks under the `ipynb` folder.

## Citing our paper

If you like this work, please cite it at
```
@article{morton2021scalable,
  title={Scalable estimation of microbial co-occurrence networks with Variational Autoencoders},
  author={Morton, J and Silverman, J and Tikhonov, G and Lahdesmaki, H and Bonneau, R},
  year={2021}
}
```
