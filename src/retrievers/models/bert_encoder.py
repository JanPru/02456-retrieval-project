import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertEncoder(nn.Module):
    def __init__(self, pooling_type='cls'):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.pooling_type = pooling_type

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        not_padding = attention_mask.unsqueeze(-1).float()   # 1 for real tokens, 0 for padding

        if self.pooling_type == 'cls':
            pooled = x[:, 0]
        elif self.pooling_type == 'mean':
            pooled = (x * not_padding).sum(dim=1) / not_padding.sum(dim=1)
        elif self.pooling_type == 'max':
            pooled = x.masked_fill(not_padding == 0, float('-inf')).max(dim=1).values

        return F.normalize(pooled, dim=-1)
