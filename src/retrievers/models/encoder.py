import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import BertConfig, BertModel

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        configuration = BertConfig(hidden_size = self.emb_dim,          # Tamany dels vectors a cada capa
                                   num_hidden_layers = self.n_layers,   # Numero de capes de transformers
                                   num_attention_heads = self.n_heads,  # Numero de attention heads per cada capa de transformer
                                   x = self.hidden_dim,                 # Mida de les q, k i v de cada head
                                   intermediate_size = self.feed_forward_dim)   # Dimensio de la capa FFN
        self.model = BertModel(configuration)
        if self.model is None:
            raise ValueError("Model not correctly initializated")

    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            vec_frase = x[:,0,:]                # Agafa nomes el primer 
            return vec_frase
            raise NotImplementedError()
        
        elif self.cfg.model.pooling_type == 'max':
            vec_frase = x.max(dim=1)            # Agafa el maxim en la dimensio 1. Dimensio 1 es la sequence_length, que es la que volem "eliminar"
            return vec_frase
            raise NotImplementedError()
        
        elif self.cfg.model.pooling_type == 'mean':
            vec_frase = x.mean(dim=1)           # Agafa la mean en la dimensio 1
            return vec_frase
            raise NotImplementedError()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        out = self.model(x)
        vec_frase = self.pool(x)
        return vec_frase
        raise NotImplementedError()
