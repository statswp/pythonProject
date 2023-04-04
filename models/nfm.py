import torch
from torch import nn

from layers.interaction import BiInteractionPooling
from models import BaseModel, DeepModel
from utils.inputs import dense_feature_to_input, sparse_feature_to_input


class NFM(BaseModel):
    def __init__(self,
                 feature_columns,
                 deep_units=(64, 32),
                 use_bn=False,
                 name='NFM'
                 ):
        super().__init__(feature_columns=feature_columns, name=name)
        self.bi_pooling = BiInteractionPooling()
        self.deep = DeepModel(in_features=self.deep_in_features, units=deep_units, use_bn=use_bn)
        self.deep_out = nn.Linear(in_features=deep_units[-1], out_features=1)

    @property
    def deep_in_features(self):
        dense_dim = sum([f.dim for f in self.dense_feature_columns])
        embed_size = self.sparse_feature_columns[0].embed_dim
        return dense_dim + embed_size

    def forward(self, x):
        dense_list = dense_feature_to_input(x, self.feature_index, self.dense_feature_columns)
        sparse_embed_list = sparse_feature_to_input(x, self.embed_dict, self.feature_index, self.sparse_feature_columns)

        bi_out = self.bi_pooling(torch.cat(sparse_embed_list, dim=1))
        dnn_inputs = torch.cat([bi_out] + dense_list, dim=1)
        out = self.deep(dnn_inputs)
        out = self.deep_out(out)
        return torch.sigmoid(out)
