import torch
import torch.nn as nn
import numpy as np

class HE2RNA(nn.Module):
    """Model that generates one score per tile and per predicted gene.

    Args
        output_dim (int): Output dimension, must match the number of genes to
            predict.
        layers (list): List of the layers' dimensions
        nonlin (torch.nn.modules.activation)
        ks (list): list of numbers of highest-scored tiles to keep in each
            channel.
        dropout (float)
        device (str): 'cpu' or 'cuda'
        mode (str): 'binary' or 'regression'
    """

    def __init__(self, input_dim, output_dim,
                 layers=[1], nonlin=nn.ReLU(), ks=[0.0125, 0.025, 0.0625, 0.125, 0.25, 0.5, 0.625],
                 dropout=0.5, device='cpu',
                 bias_init=None, **kwargs):
        super(HE2RNA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [input_dim] + layers + [output_dim]
        self.layers = []
        for i in range(len(layers) - 1):
            layer = nn.Conv1d(in_channels=layers[i],
                              out_channels=layers[i+1],
                              kernel_size=1,
                              stride=1,
                              bias=True)
            setattr(self, 'conv' + str(i), layer)
            self.layers.append(layer)
        if bias_init is not None:
            self.layers[-1].bias = bias_init
        self.ks = np.array(ks)

        self.nonlin = nonlin
        self.do = nn.Dropout(dropout)
        self.device = device
        self.to(self.device)

    def forward(self, x, training=False):
        with torch.no_grad():
            x.transpose_(1, 2)
            k = int(np.random.choice(self.ks) * x.shape[2])
        if training:
            return self.forward_fixed_k(x, k), k
        else:
            pred = 0
            for k in self.ks:
                k = k * x.shape[2]
                pred += self.forward_fixed_k(x, int(k)) / len(self.ks)
            return pred, 0

    def forward_fixed_k(self, x, k):
        torch.cuda.empty_cache()
        mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = (mask > 0).float()
        x = self.conv(x) * mask
        x = torch.sum(torch.topk(x, k, dim=2, largest=True, sorted=True)[0] * mask[:, :, :k], dim=2) / torch.sum(mask[:, :, :k], dim=2)
        return x

    def conv(self, x):
        x = x[:, x.shape[1] - self.input_dim:]
        for i in range(len(self.layers) - 1):
            x = self.do(self.nonlin(self.layers[i](x)))
        x = self.layers[-1](x)
        return x
