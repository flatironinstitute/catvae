# catvae
Categorical Variational Autoencoders

# What are Categorical Variational Autoencoders?

Variational Autoencoders (VAE) are one of the state-of-the-art methods applying neural networks to perform Bayesian inference to estimate complex high dimensional distributions, with recent techniques showing that Linear VAEs are mathematically equivalent to Principal Components Analysis.


Categorical Variational Autoencoders or Multinomial Variational Autoencoders are extentions of VAEs applied to count data.  These methods can estimate the moments of the Multinomial Logistic Normal on distribution with thousands of dimensions and thousands of observations in the matter of hours.

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

## Downloading models

The pretrained Mouse VAE can be found [here](https://users.flatironinstitute.org/jmorton/public_www/catvae-mouse-z128-l5-deblur.tar.gz), and can be downloaded via
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae-mouse-z128-l5-deblur.tar.gz
tar -zxvf catvae-mouse-z128-l5-deblur.tar.gz
```

## Loading models

When processing your own data, it is important to note that you can only perform inference on the microbes that have been observed by the VAE.  As a result, it is critical that your data is completely aligned with the VAE. Loading the VAE model and aligning your data against the VAE can be done as follows

```python
import torch
import biom
from skbio
from gneiss.util import match_tips

# Load model files
vae_model_path = 'catvae-mouse-z128-l5-deblur'
ckpt_path = os.path.join(vae_model_path, 'last_ckpt.pt')
params = os.path.join(vae_model_path, 'hparams.yaml')    
nwk_path = os.path.join(vae_model_path, 'tree.nwk')  
tree = skbio.TreeNode(nwk_path)
with open(params, 'r') as stream:   
    params = yaml.safe_load(stream)     
params['basis'] = nwk_path
vae_model = MultVAE.load_from_checkpoint(ckpt_path, **params)

# Load your dataset
X_train = biom.load_table('<your biom table>')
# Align your data against the VAE
X_train, tree = match_tips(X_train, tree)
```

If you want to obtain a reduced dimension representation of your data, that can be done as follows
```python
# Convert pandas dataframe to numpy array
X_train = X_train.to_dataframe().to_dense().values
# Obtain dimensionality reduced data
X_embed = vae_model.to_latent(
        torch.Tensor(X_train).float()).detach().cpu().numpy()
```

You can also sample from these embeddings. Below is an example of how you would do that.
```python
x = X_train[0, :]
vae_model.sample(x)
```

If you want to obtain a CLR representation of the VAE decoder loadings, it can be done as follows
```python
import pandas as pd
from gneiss.balances import sparse_balance_basis
Psi, int_nodes = sparse_balance_basis(tree)
# ILR representation of the VAE decoder loadings
W = vae_model.vae.decoder.weight.detach().numpy()
# CLR representation of the VAE decoder loadings
names = [n.name for n in tree.tips()]
cW = pd.DataFrame(Psi.T @ W, index=names)
```

## Training the VAE models

Please refer to the Jupyter notebooks under the `ipynb` folder.
