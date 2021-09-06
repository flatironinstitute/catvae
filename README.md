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

## Downloading pretrained models

[Pretrained Mouse VAE 128 latent dimensions](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-mouse-z128-l5-deblur.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae-mouse-z128-l5-deblur.tar.gz
tar -zxvf catvae-mouse-z128-l5-deblur.tar.gz
```
[Pretrained Batch corrected Mouse VAE 128 latent dimensions](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-mouse-z128-l5-deblur-batch.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae-mouse-z128-l5-deblur-batch.tar.gz
tar -zxvf catvae-mouse-z128-l5-deblur-batch.tar.gz
```
[Pretrained Human VAE 128 latent dimensions](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-human-z128-l5-deblur.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae-human-z128-l5-deblur.tar.gz
tar -zxvf catvae-human-z128-l5-deblur-batch.tar.gz
```
[Pretrained Batch corrected Human VAE 128 latent dimensions](https://users.flatironinstitute.org/jmorton/public_www/catvae_models/catvae-human-z128-l5-deblur-batch.tar.gz)
```
wget https://users.flatironinstitute.org/jmorton/public_www/catvae-human-z128-l5-deblur-batch.tar.gz
tar -zxvf catvae-human-z128-l5-deblur-batch.tar.gz
```
## Pre processing your data

All of the pretrained models were trained on 100bp 16S V4 deblurred data from [Qiita](https://qiita.ucsd.edu/).  To use these models on your data, either upload your data to Qiita, or process your data using deblur.  See the [qiime2 tutorial](https://docs.qiime2.org/2021.4/tutorials/moving-pictures/#option-2-deblur) for an example of how to deblur your amplicon data.

## Loading VAE models

When processing your own data, it is important to note that you can only perform inference on the microbes that have been observed by the VAE.  As a result, it is critical that your data is completely aligned with the VAE. Loading the VAE model and aligning your data against the VAE can be done as follows

```python
import torch
import biom
from skbio
from gneiss.util import match_tips
from catvae.trainer import MultVAE 

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

## Loading Batch corrected VAE models

The process is almost identical
```python
import torch
import biom
from skbio
from gneiss.util import match_tips
from catvae.trainer import MultBatchVAE 

# Load model files
vae_model_path = 'catvae-mouse-z128-l5-deblur-batch'
ckpt_path = os.path.join(vae_model_path, 'last_ckpt.pt')
params = os.path.join(vae_model_path, 'hparams.yaml')    
nwk_path = os.path.join(vae_model_path, 'tree.nwk')  
tree = skbio.TreeNode(nwk_path)
with open(params, 'r') as stream:   
    params = yaml.safe_load(stream)     
params['basis'] = nwk_path
vae_model = MultBatchVAE.load_from_checkpoint(ckpt_path, **params)

# Load your dataset
X_train = biom.load_table('<your biom table>')
# Align your data against the VAE
X_train, tree = match_tips(X_train, tree)
```
Extracting latent representations and sampling is slightly different since the batch information needs to be specified.
All of the batch names are under the `batch_categories.txt` file, but the model only takes numerical ids as shown in the first column.
```python
batch_num = <your specified batch>
X_embed = vae_model.to_latent(
        torch.Tensor(X_train).float(), batch_num).detach().cpu().numpy()
        
x = X_train[0, :]
vae_model.sample(x, batch_num)
```

## Training the VAE models

Please refer to the Jupyter notebooks under the `ipynb` folder.
