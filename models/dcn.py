import torch
from torch import nn

from layers import CrossLayer
from models import BaseModel, DeepModel
from utils.inputs import dense_feature_to_input, sparse_feature_to_input, varlen_feature_to_inputs


class DCN(BaseModel):
    def __init__(self,
                 feature_columns,
                 deep_units=(64, 32),
                 use_bn=False,
                 cross_type="vector",
                 name="DCN"
                 ):
        super().__init__(feature_columns=feature_columns, name=name)

        self.in_features = self.compute_in_features()
        self.deep = DeepModel(in_features=self.in_features, units=deep_units, use_bn=use_bn)
        self.cross_layer = CrossLayer(in_features=self.in_features, cross_type=cross_type)
        self.linear = nn.LazyLinear(out_features=1)

    def forward(self, x):
        dense_inputs = dense_feature_to_input(x, self.feature_index,
                                            self.dense_feature_columns)  # (B, K)
        sparse_inputs = sparse_feature_to_input(x, self.embed_dict, self.feature_index,
                                                self.sparse_feature_columns)  # list of (B, 1, E)
        varlen_inputs = varlen_feature_to_inputs(x, self.embed_dict, self.feature_index,
                                                 self.varlen_feature_columns, reduction=True)  # list of (B, 1, E)

        sparse_inputs_flatten = [torch.flatten(t, start_dim=1) for t in sparse_inputs]
        inputs = torch.cat(dense_inputs + sparse_inputs_flatten + varlen_inputs, dim=1)
        cross_out = self.cross_layer(inputs)
        deep_out = self.deep(inputs)
        cat_out = torch.cat([cross_out, deep_out], dim=1)

        logit = self.linear(cat_out)
        return torch.sigmoid(logit)
