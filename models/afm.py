import torch
from torch import nn

from layers import AFMLayer
from models import BaseModel, DeepModel
from utils.inputs import dense_feature_to_input, sparse_feature_to_input


class AFM(BaseModel):
    def __init__(self,
                 feature_columns,
                 deep_units,
                 embed_size,
                 atten_factor,
                 use_bn=False,
                 name='AFM'
                 ):
        super().__init__(feature_columns=feature_columns, name=name)
        self.afm_layer = AFMLayer(embed_size=embed_size, atten_factor=atten_factor)
        self.deep = DeepModel(in_features=self.compute_in_features(), units=deep_units, use_bn=use_bn)
        self.deep_out = nn.LazyLinear(out_features=1)

    def forward(self, x):
        dense_inputs = dense_feature_to_input(x, self.feature_index,
                                              self.dense_feature_columns)
        sparse_inputs = sparse_feature_to_input(x, self.embed_dict, self.feature_index,
                                                self.sparse_feature_columns)
        # varlen_inputs = varlen_feature_to_inputs(x, self.embed_dict, self.feature_index,
        #                                          self.varlen_feature_columns)

        deep_inputs = torch.cat(dense_inputs + [torch.flatten(t, start_dim=1) for t in sparse_inputs], dim=1)
        deep_out = self.deep(deep_inputs)
        deep_logit = self.deep_out(deep_out)

        afm_logit = self.afm_layer(sparse_inputs)

        logit = deep_logit + afm_logit
        return torch.sigmoid(logit)
