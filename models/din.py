from layers import AttentionSequencePoolingLayer
from models import BaseModel, DeepModel
from utils.inputs import *


class DIN(BaseModel):
    def __init__(self,
                 feature_columns,
                 deep_units,
                 atten_embedding_dim,
                 weight_normalization=False,
                 use_bn=False,
                 name='DIN'):
        super().__init__(feature_columns=feature_columns, name=name)
        self.atten_embedding_dim = atten_embedding_dim
        self.attn_layer = AttentionSequencePoolingLayer(atten_embedding_dim=atten_embedding_dim,
                                                        atten_use_bn=True,
                                                        weight_normalization=weight_normalization,
                                                        return_score=False
                                                        )
        self.deep = DeepModel(in_features=self.in_features, units=deep_units, use_bn=use_bn)
        self.deep_out = nn.LazyLinear(1)

    @property
    def in_features(self):
        return self.atten_embedding_dim

    def forward(self, x):
        dense_inputs = dense_feature_to_input(x, self.feature_index,
                                              self.dense_feature_columns, return_dict=True)
        sparse_inputs = sparse_feature_to_input(x, self.embed_dict, self.feature_index,
                                                self.sparse_feature_columns, return_dict=True)
        varlen_inputs = varlen_feature_to_inputs(x, self.embed_dict, self.feature_index,
                                                 self.varlen_feature_columns, return_dict=True)

        query = torch.cat([
            sparse_inputs['target_id'],
            sparse_inputs['target_cate']
        ], dim=-1)

        keys = torch.cat([
            varlen_inputs['item_ids'],
            varlen_inputs['cate_ids']
        ], dim=-1)

        keys_len = dense_inputs.pop('item_ids_length')

        atten_out = self.attn_layer(query, keys, keys_len)  # (B, 1, E)

        dnn_input = torch.flatten(atten_out, start_dim=1)
        dnn_out = self.deep(dnn_input)
        dnn_logit = self.deep_out(dnn_out)

        out = torch.sigmoid(dnn_logit)
        return out
