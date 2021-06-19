import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TripletNet(nn.Module):

    def __init__(self, input_size, embed_dim):
        """ Initialize model parameters.

        Parameters
        ----------
        input_size: int
            Input dimension size
        embed_dim: int
            Embedding dimention, typically from 50 to 500.

        Notes
        -----
        The language_model must be a subclass of torch.nn.Module
        and must also have an `extract_features` method, which takes
        in as input a peptide encoding an outputs a latent representation.
        """
        super(TripletNet, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.embeddings = nn.Linear(input_size, embed_dim)
        self.init_emb()

    def init_emb(self):
        initstd = 1 / math.sqrt(self.embed_dim)
        self.embeddings.weight.data.normal_(0, initstd)
        self.embeddings.weight.data.normal_(0, initstd)

    def encode(self, x):
        return self.embeddings(x)

    def forward(self, pos_u, pos_v, neg_v):
        """ Obtains triples and computes triplet score

        Parameters
        ----------
        pos_u : torch.Tensor
            Reference latent vector
        pos_v : torch.Tensor
            Positive latent vector
        neg_v : torch.Tensor
            Negative latent vector
        """
        losses = 0
        emb_u = self.embeddings(pos_u)
        emb_v = self.embeddings(pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, -1)
        score = F.logsigmoid(score)
        if score.dim() >= 1:
            losses += sum(score)
        else:
            losses += score
        neg_emb_v = self.embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v.unsqueeze(1),
                              emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        if neg_score.dim() >= 1:
            losses += sum(neg_score)
        else:
            losses += neg_score
        return -losses
