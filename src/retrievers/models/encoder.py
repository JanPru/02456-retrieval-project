import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None # store the attention maps
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout = 0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ffnn, dropout = 0.1):
        super(FeedForward,self).__init__()
        self.l1 = nn.Linear(d_model,d_ffnn) # first layer
        self.relu = nn.ReLU() # activation function
        self.l2 = nn.Linear(d_ffnn,d_model) # second layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        return out

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))

class TransformerLayer(nn.Module):
    def __init__(self, h, d_model, d_ffnn, dropout = 0.1):
        super(TransformerLayer,self).__init__()
        self.mha = MultiHeadedAttention(h, d_model, dropout)
        self.ffnn = FeedForward(d_model, d_ffnn, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask = None):
        x = self.addnorm1(x, self.mha(x, x, x, mask))
        x = self.addnorm2(x, self.ffnn(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, vocab_size: int = 30522, pooling_type: str = 'cls'):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.vocab_size = vocab_size
        self.pooling_type = pooling_type
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_enc = PositionalEncoding(self.emb_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(self.n_heads, self.emb_dim, self.feed_forward_dim) for _ in range(self.n_layers)
        ])

    def pool(self, x, mask = None):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """

        not_padding = mask.squeeze(1).unsqueeze(-1).float()

        if self.pooling_type == 'cls':
            vec_frase = x[:,0,:]                # Agafa nomes el primer 
            return vec_frase
        
        elif self.pooling_type == 'max':
            x = x.masked_fill(not_padding == 0, float('-inf'))
            vec_frase = x.max(dim=1).values            # Agafa el maxim en la dimensio 1. Dimensio 1 es la sequence_length, que es la que volem "eliminar"
            return vec_frase
        
        elif self.pooling_type == 'mean':
            suma = (x * not_padding).sum(dim=1)
            reals = not_padding.sum(1).clamp(min=1)
            vec_frase = suma/reals      # Agafa la mean en la dimensio 1. S'ha de fer la masked attention???
            return vec_frase

    def forward(self, x: torch.tensor, attention_mask) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        mask = attention_mask.unsqueeze(1)
        # Ens entra documents o querys
        x = self.pos_enc(self.embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return F.normalize(self.pool(x,mask), dim=-1)

    