from itertools import combinations

import torch
from torch import nn


class FM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # (B, K, E)
        square_of_sum = x.sum(dim=1) ** 2  # (B, E)
        sum_of_square = (x ** 2).sum(dim=1)  # (B, E)
        cross_out = (square_of_sum - sum_of_square).sum(dim=1, keepdim=True) * 0.5  # (B, 1)
        return cross_out  # (B, 1)


class BiInteractionPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, K, E)
        square_of_sum = x.sum(dim=1) ** 2  # (B, E)
        sum_of_square = (x ** 2).sum(dim=1)  # (B, E)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term  # (B, E)


class CrossLayer(nn.Module):
    def __init__(
            self,
            in_features,
            num_layers=2,
            cross_type="vector",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.cross_type = cross_type

        if self.cross_type == "vector":
            self.kernels = nn.Parameter(torch.Tensor(self.num_layers, in_features, 1))
        elif self.cross_type == "matrix":
            self.kernels = nn.Parameter(torch.Tensor(self.num_layers, in_features, in_features))

        self.bias = nn.Parameter(torch.Tensor(num_layers, in_features, 1))

        self.init_weights()

    def init_weights(self):
        for i in range(self.kernels.size(0)):
            nn.init.xavier_uniform_(self.kernels[i])
        for i in range(self.bias.size(0)):
            nn.init.xavier_uniform_(self.bias[i])

    def forward(self, x: torch.Tensor):
        x0 = x.unsqueeze(dim=2)  # (B,E,1)
        xl = x0

        for i in range(self.num_layers):
            if self.cross_type == "vector":
                w = self.kernels[i]  # (E,1)
                b = self.bias[i]  # (E,1)
                xl_w = torch.matmul(torch.transpose(xl, 1, 2), w)  # (B,1,1)
                x0_l = torch.matmul(x0, xl_w)  # (B,E,1)
                x0_l = x0_l + b  # (B,E,1)
                xl = x0_l + xl
            elif self.cross_type == "matrix":
                w = self.kernels[i]  # (E,E)
                b = self.bias[i]  # (E,1)
                xl_w = torch.matmul(w, xl)  # (B,E,1)
                dot_ = xl_w + b  # (B,E,1)
                xl = x0 * dot_ + xl  # (B,E,1)

        return xl.squeeze(dim=2)


class InnerProductLayer(nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        # list of (B, 1, E)
        n_fields = len(x)
        indicies = list(combinations(range(n_fields), 2))

        p = torch.cat([x[i] for i, _ in indicies], dim=1)
        q = torch.cat([x[j] for _, j in indicies], dim=1)
        inner_product = p * q  # (B, K(K-1)/2, E)

        if self.reduce_sum:
            inner_product = torch.sum(inner_product, dim=-1)  # # (B, K(K-1)/2)

        return inner_product


class AFMLayer(nn.Module):
    def __init__(self,
                 embed_size,
                 atten_factor
                 ):
        super().__init__()
        self.atten_factor = atten_factor
        self.atten_weight = nn.Parameter(torch.Tensor(embed_size, self.atten_factor))
        self.atten_bias = nn.Parameter(torch.Tensor(self.atten_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1))
        self.projection_p = nn.Parameter(torch.Tensor(embed_size, 1))

    def forward(self, x):
        """
        :param x: list of (B, 1, E)
        :return:
        """
        row = []
        col = []
        for i, j in combinations(x, 2):
            row.append(i)
            col.append(j)
        p = torch.cat(row, dim=1)  # (B, N(N-1)/2, E)
        q = torch.cat(col, dim=1)
        inner_product = p * q  # (B, N(N-1)/2, E)

        atten_score = torch.matmul(inner_product, self.atten_weight) + self.atten_bias
        atten_score = torch.relu(atten_score)  # (B, N(N-1)/2, atten_factor)

        atten_score = torch.matmul(atten_score, self.projection_h)  # (B, N(N-1)/2, 1)
        norm_atten_score = torch.softmax(atten_score, dim=1)  # (B, N(N-1)/2, 1)

        atten_out = torch.sum(norm_atten_score * inner_product, dim=1)  # (B, E)
        out = torch.matmul(atten_out, self.projection_p)  # (B, 1)
        return out


if __name__ == "__main__":
    # x = torch.randn(32, 8)
    # # layer = CrossLayer(in_features=8, cross_type="matrix")
    # layer = CrossLayer(in_features=8, cross_type="vector")
    # out = layer(x)
    # print(out.shape)

    x = torch.rand(4, 3, 5)

    layer = BiInteractionPooling()
    out = layer(x)
    print(out)

    x1 = torch.split(x, 1, dim=1)
    layer1 = InnerProductLayer(reduce_sum=True)
    out1 = layer1(x1)
    print(out1)
