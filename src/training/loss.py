import torch
import torch.nn as nn
import torch.nn.functional as F

class DPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        Computes cross entropy between the query 

        Returns: A 1d torch tensor 
        """
        score = queries @ documents.T
        labels = torch.arange(queries.shape[0], device=score.device)
        return F.cross_entropy(score, labels)