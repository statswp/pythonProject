from collections import OrderedDict
from typing import List

import torch
from torch import nn


class Feature:
    def __init__(self, name):
        self.name = name


class DenseFeature(Feature):
    def __init__(self, name, dim=1):
        super().__init__(name=name)
        self.dim = dim

    def __str__(self):
        return f'Name: {self.name}, dim: {self.dim}'


class SparseFeature(Feature):
    def __init__(self, name, vocab_size, embed_dim=8, embed_name=None, dim=1):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_name = embed_name if embed_name else name
        self.dim = dim

    def __str__(self):
        return f"Name: {self.name}, vocab_size: {self.vocab_size}, " \
               f"embed_dim: {self.embed_dim}, embed_name: {self.embed_name}"


class VarlenFeature(SparseFeature):
    def __init__(self, name, vocab_size, embed_dim=8, embed_name=None, max_len=10, reduction='mean'):
        super().__init__(name, vocab_size, embed_dim, embed_name, dim=1)
        self.max_len = max_len
        self.reduction = reduction


def create_embed_dict(feature_columns):
    embed_dict = nn.ModuleDict()
    sparse_feature_columns = [f for f in feature_columns if isinstance(f, SparseFeature)]
    varlen_feature_columns = [f for f in feature_columns if isinstance(f, VarlenFeature)]

    for f in sparse_feature_columns + varlen_feature_columns:
        embed_name = f.embed_name
        if embed_name in embed_dict:
            continue
        embed_table = nn.Embedding(num_embeddings=f.vocab_size, embedding_dim=f.embed_dim)
        nn.init.xavier_uniform_(embed_table.weight)
        embed_dict[embed_name] = embed_table
    return embed_dict


def create_feature_index(feature_columns):
    feature_index = {}
    start = 0
    for f in feature_columns:
        if isinstance(f, VarlenFeature):
            feature_index[f.name] = [start, start + f.max_len]
            start = start + f.max_len
        elif isinstance(f, (DenseFeature, SparseFeature)):
            feature_index[f.name] = [start, start + f.dim]
            start = start + f.dim
    return feature_index


def sparse_feature_to_input(x, embed_dict, feature_index, sparse_feature_columns: List[SparseFeature],
                            return_dict=False):
    sparse_vector_dict = OrderedDict()
    for f in sparse_feature_columns:
        embed_name = f.embed_name
        embed_table = embed_dict[embed_name]
        feat_name = f.name
        indices = x[:, feature_index[feat_name][0]:feature_index[feat_name][1]].long()
        embed = embed_table(indices)
        sparse_vector_dict[feat_name] = embed
    if return_dict:
        return sparse_vector_dict
    return list(sparse_vector_dict.values())


def varlen_feature_to_inputs(x, embed_dict, feature_index, varlen_feature_columns: List[VarlenFeature],
                             reduction=False, return_dict=False):
    varlen_vector_dict = OrderedDict()
    for f in varlen_feature_columns:
        feat_name = f.name
        embed_name = f.embed_name
        embed_table = embed_dict[embed_name]
        indices = x[:, feature_index[feat_name][0]:feature_index[feat_name][1]].long()
        embed = embed_table(indices)

        if reduction:
            if f.reduction == 'mean':
                varlen_vector_dict[feat_name] = torch.mean(embed, dim=1, keepdim=True)
            elif f.reduction == 'sum':
                varlen_vector_dict[feat_name] = torch.mean(embed, dim=1, keepdim=True)
            else:
                raise ValueError('f.reduction is None')
        else:
            varlen_vector_dict[feat_name] = embed

    if return_dict:
        return varlen_vector_dict
    return list(varlen_vector_dict.values())


def dense_feature_to_input(x, feature_index, dense_feature_list: List[DenseFeature], return_dict=False):
    dense_list = OrderedDict()
    for f in dense_feature_list:
        feat_name = f.name
        dense_list[feat_name] = x[:, feature_index[feat_name][0]: feature_index[feat_name][1]]

    if return_dict:
        return dense_list
    return list(dense_list.values())
