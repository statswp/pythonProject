import torch

from layers.interaction import InnerProductLayer
from models import BaseModel, DeepModel
from utils.inputs import *


class PNN(BaseModel):
    def __init__(self,
                 feature_columns,
                 deep_units=(64, 32),
                 use_bn=False,
                 reduce_sum=True,
                 name='PNN'
                 ):
        super().__init__(feature_columns=feature_columns, name=name)
        self.inner_product = InnerProductLayer(reduce_sum=reduce_sum)
        self.deep = DeepModel(in_features=self.deep_in_features, units=deep_units, use_bn=use_bn)
        self.deep_out = nn.Linear(in_features=deep_units[-1], out_features=1)

    @property
    def deep_in_features(self):
        in_features = self.compute_in_features()
        n_fields = len(self.sparse_feature_columns)
        return in_features + int(n_fields * (n_fields - 1) * 0.5)

    def forward(self, x):
        dense_list = dense_feature_to_input(x, self.feature_index, self.dense_feature_columns)
        sparse_embed_list = sparse_feature_to_input(x, self.embed_dict, self.feature_index, self.sparse_feature_columns)
        sparse_embed_inputs = torch.cat([torch.flatten(t, start_dim=1) for t in sparse_embed_list], dim=1)

        inner_product = self.inner_product(sparse_embed_list)
        dnn_inputs = torch.cat(dense_list, dim=1)
        dnn_inputs = torch.cat([dnn_inputs, sparse_embed_inputs, inner_product], dim=1)

        out = self.deep(dnn_inputs)
        logit = self.deep_out(out)
        return torch.sigmoid(logit)
