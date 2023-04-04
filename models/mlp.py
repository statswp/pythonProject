from torch import nn


class DeepModel(nn.Module):
    def __init__(self,
                 in_features=None,
                 units=(64, 32),
                 use_bn=False,
                 ):
        super().__init__()
        self.deep = nn.Sequential()
        for unit in units:
            linear_layer = nn.Linear(in_features, out_features=unit)
            nn.init.xavier_uniform_(linear_layer.weight)
            self.deep.append(linear_layer)

            if use_bn:
                self.deep.append(nn.LazyBatchNorm1d())

            self.deep.append(nn.ReLU())
            in_features = unit

    def forward(self, x):
        out = self.deep(x)
        return out
