import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, bn=None):
        super(DeepMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if bn is not None:
            layers.append(nn.BatchNorm1d(bn))
        layers.append(nn.GELU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if bn is not None:
                layers.append(nn.BatchNorm1d(bn))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        if bn is not None:
            layers.append(nn.BatchNorm1d(bn))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class DeepMLP_g(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepMLP_g, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.Dmlp_r = DeepMLP_g(256, 256)
        self.Dmlp_n = DeepMLP_g(256, 256)

    def forward(self, r, n):
        n[:, 18, :] += r[:, 20, :]
        n[:, 18, :] += r[:, 19, :]
        r[:, 17, :] += n[:, 18, :]

        n[:, 16, :] += r[:, 16, :] + r[:, 17, :] + r[:, 18, :]
        r[:, 15, :] += n[:, 16, :]

        n[:, 14, :] += r[:, 14, :] + r[:, 15, :] + r[:, 16, :]
        r[:, 13, :] += n[:, 14, :]

        n[:, 11, :] += r[:, 13, :] + r[:, 14, :]
        r[:, 10, :] += n[:, 11, :]

        n[:, 10, :] += r[:, 12, :] + r[:, 11, :]
        r[:, 9, :] += n[:, 10, :]

        n[:, 8, :] += r[:, 10, :] + r[:, 9, :]
        r[:, 7, :] += n[:, 8, :]

        n[:, 7, :] += r[:, 7, :] + r[:, 8, :]
        r[:, 6, :] += n[:, 7, :]

        n[:, 4, :] += r[:, 5, :] + r[:, 6, :]
        r[:, 3, :] += n[:, 4, :]

        n[:, 1, :] += r[:, 0, :] + r[:, 2, :]
        r[:, 1, :] += n[:, 1, :]

        n[:, 2, :] += r[:, 1, :] + r[:, 3, :]
        r[:, 4, :] += n[:, 2, :]

        n[:, 5, :] += r[:, 4, :]

        r = self.Dmlp_r(r)
        n = self.Dmlp_n(n)
        return r, n


class SimpleTransformerModel(nn.Module):
    def __init__(self, Sequence_len, feature_dim, nhead, num_encoder_layers, dim_feedforward):
        super(SimpleTransformerModel, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, Sequence_len, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src):
        src = src + self.positional_encoding
        output = self.transformer_encoder(src)
        return output


class FeatureCompressor(nn.Module):
    def __init__(self, input_features, output_features, feature_dim):

        super(FeatureCompressor, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.feature_dim = feature_dim

        self.fc = nn.Linear(input_features, output_features)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.input_features)
        x = self.fc(x)
        x = x.reshape(-1, self.feature_dim, self.output_features)
        x = x.transpose(1, 2)
        return x

class FeatureOUT(nn.Module):
    def __init__(self, input_features, output_features, feature_dim):

        super(FeatureOUT, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(input_features, feature_dim)
        self.fc2 = nn.Linear(feature_dim, output_features)
        self.gelu = nn.GELU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.gelu(x)

        return x


class HTGNN_WNP(nn.Module):
    def __init__(self):
        super(HTGNN_WNP, self).__init__()

        self.MLP = DeepMLP(18, 128, [64], bn=10)
        self.transformer_v = SimpleTransformerModel(Sequence_len=128, feature_dim=10,
                                                    nhead=1, num_encoder_layers=1, dim_feedforward=512)
        self.transformer_s = SimpleTransformerModel(Sequence_len=10, feature_dim=128,
                                                  nhead=1, num_encoder_layers=1, dim_feedforward=512)
        self.fc = FeatureCompressor(10, 1, 128)
        self.Dmlp_r = DeepMLP(12, 128, [64], bn=21)
        self.Dmlp_n = DeepMLP(3, 256, [128], bn=22)
        self.transformer_r = SimpleTransformerModel(Sequence_len=21, feature_dim=128,
                                                   nhead=1, num_encoder_layers=1, dim_feedforward=512)
        self.gnn = GNN()
        self.fc_out = FeatureCompressor(43, 1, 256)

        self.out = nn.Linear(256, 1)

    def forward(self, x_r, x_g, x_n):
        batch_size = x_g.shape[0]

        # Hierarchical Transformer for Time-series Modelling
        x_g = self.MLP(x_g)
        x_g = x_g.transpose(1, 2)
        x_g = self.transformer_v(x_g).transpose(1, 2)
        x_g = self.transformer_s(x_g)
        x_g = self.fc(x_g)

        # Watershed Hydrological Modeling
        x_r = self.Dmlp_r(x_r)
        x_r = self.transformer_r(x_r)
        # Broadcast of global features.
        x_g = x_g.expand(batch_size, 21, 128)
        x_r = torch.cat((x_r, x_g), dim=-1)
        # Node feature extraction
        x_n = self.Dmlp_n(x_n)

        # Watershed Topology Modeling
        for _ in range(1):
            x_r, x_n = self.gnn(x_r, x_n)

        out = torch.cat((x_r, x_n), dim=1)
        fc_out = self.fc_out(out)
        out = self.out(fc_out)

        return out, fc_out


if __name__ == '__main__':
    x_r = torch.randn(100, 21, 12)
    x_g = torch.randn(100, 10, 18)
    x_n = torch.randn(100, 22, 3)
    net = HTGNN_WNP()
    out, fc_out = net(x_r, x_g, x_n)
    print(out.shape)






