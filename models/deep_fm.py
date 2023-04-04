import torch
from torch import nn
from models import BaseModel, DeepModel
from layers import FM
from utils.inputs import dense_feature_to_input, sparse_feature_to_input


class DeepFM(BaseModel):
    def __init__(self,
                 feature_columns,
                 deep_units=(64, 32),
                 use_bn=False,
                 name='DeepFM'
                 ):
        super(DeepFM, self).__init__(feature_columns=feature_columns, name=name)
        self.in_features = self.compute_in_features()
        self.fm = FM()
        self.deep = DeepModel(in_features=self.in_features, units=deep_units, use_bn=use_bn)
        self.deep_out = nn.LazyLinear(1)

    def forward(self, x):
        # (B, K)
        dense_list = dense_feature_to_input(x, self.feature_index, self.dense_feature_columns)
        # list of (B, 1, E)
        sparse_embed_list = sparse_feature_to_input(x, self.embed_dict, self.feature_index, self.sparse_feature_columns)

        fm_inputs = torch.cat(sparse_embed_list, dim=1)
        fm_logit = self.fm(fm_inputs)

        dnn_input = torch.cat([torch.flatten(t, start_dim=1) for t in sparse_embed_list] + dense_list, dim=1)
        dnn_out = self.deep(dnn_input)
        dnn_logit = self.deep_out(dnn_out)

        out = torch.sigmoid(fm_logit + dnn_logit)
        return out
