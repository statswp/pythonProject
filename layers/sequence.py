import torch
from torch import nn

from models import DeepModel


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_units=(64, 32), atten_embedding_dim=8, use_bn=False):
        super().__init__()
        self.deep = DeepModel(in_features=4 * atten_embedding_dim, units=hidden_units, use_bn=use_bn)
        self.deep_out = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, keys, keys_len):
        """
        :param query: (B, 1, E)
        :param keys: (B, T, E)
        :param keys_len: (B, T, E)
        :return:
        """
        seq_len = keys.size(1)
        query = query.expand(-1, seq_len, -1)  # (B, T, E)
        atten_inputs = torch.cat([
            query,
            keys,
            query - keys,
            query * keys
        ], dim=-1)
        atten_out = self.deep(atten_inputs)
        atten_score = self.deep_out(atten_out)
        return atten_score


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self,
                 atten_units=(64, 32),
                 atten_embedding_dim=8,
                 atten_use_bn=False,
                 weight_normalization=False,
                 return_score=False,
                 ):
        super().__init__()
        self.local_atten = LocalActivationUnit(hidden_units=atten_units,
                                               atten_embedding_dim=atten_embedding_dim,
                                               use_bn=atten_use_bn)
        self.weight_normalization = weight_normalization
        self.return_score = return_score

    def forward(self, query, keys, keys_len):
        """
        :param query: (B, 1, E)
        :param keys: (B, T, E)
        :param keys_len: (B, 1)
        :return:
        """
        batch_size, keys_max_len, embed_size = keys.size()

        attention_score = self.local_atten(query, keys, keys_len)  # (B, T, 1)
        attention_score = torch.transpose(attention_score, 1, 2)  # (B, 1, T)

        keys_max_len = torch.arange(keys_max_len, dtype=keys_len.dtype).repeat(batch_size, 1)  # (B, T)
        keys_mask = keys_max_len < keys_len.view(-1, 1)  # (B, T)
        keys_mask = keys_mask.unsqueeze(dim=1)  # (B, 1, T)

        paddings = torch.zeros_like(attention_score)  # (B, 1, T)
        if self.weight_normalization:
            paddings = torch.ones_like(attention_score) * (-(2 ** 32) + 1)

        scores = torch.where(keys_mask, attention_score, paddings)  # (B, 1, T)
        if self.weight_normalization:
            scores = torch.softmax(scores, dim=-1)

        if self.return_score:
            return scores
        return torch.matmul(scores, keys)
