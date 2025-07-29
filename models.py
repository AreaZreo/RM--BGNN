import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x




class RMBGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, use_bn=False):
        super(RMBGNN, self).__init__()
        self.pre = BayesianLinear(nfeat, nhid)
        self.post = BayesianLinear(nhid, nclass)
        self.proj_head1 = Linear(nhid, nhid, dropout, bias=True)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid)

    # 不用MLP_encoder

    # cora: AVG_ACC:68.37+-3.89（残差0.01）hidden=32 trail=5 T=20 train_per_class 23 num_neighbor 15
    # cora: AVG_ACC:67.58+-3.36（残差0.01）hidden=32 trail=5 T=20 train_per_class 23 num_neighbor 20

    def forward(self, features, eval=False):
        features = features.to(device)  # 确保 features 在正确的设备上
        if self.use_bn:
            features = self.bn1(features)
        if self.use_bn:
            features = self.bn2(features)
        query_features2 = self.pre(features)
        out = self.post(query_features2)
        if not eval:
            emb = self.proj_head1(query_features2)
        else:
            emb = None
        return emb, F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with normal prior and normal posterior
    """
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.log_prior = 0.0
        self.log_variational_posterior = 0.0

    def forward(self, x):
        weight_epsilon = torch.randn(self.out_features, self.in_features).to(x.device)
        bias_epsilon = torch.randn(self.out_features).to(x.device)
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_std * weight_epsilon
        bias = self.bias_mu + bias_std * bias_epsilon

        # 权重和偏置的总对数先验概率

        self.log_prior = normal_logpdf(weight, torch.tensor(0.0, device=x.device), torch.tensor(1.0, device=x.device)).sum() + normal_logpdf(bias, torch.tensor(0.0, device=x.device), torch.tensor(1.0, device=x.device)).sum()
        self.log_variational_posterior = normal_logpdf(weight, self.weight_mu, weight_std).sum() + normal_logpdf(bias, self.bias_mu, bias_std).sum()

        return F.linear(x, weight, bias)


#normal_logpdf 函数是用于计算给定均值 mu 和标准差 sigma 的正态分布在 x 处的对数概率密度值。
def normal_logpdf(x, mu, sigma):
    """
    Computes the log pdf of a normal distribution with mean mu and variance sigma at x
    """
    device = x.device  # 确保 mu 和 sigma 在与 x 相同的设备上
    mu = torch.tensor(mu, device=device) if not isinstance(mu, torch.Tensor) else mu
    sigma = torch.tensor(sigma, device=device) if not isinstance(sigma, torch.Tensor) else sigma
    c = torch.tensor(2.0 * math.pi, device=device)
    return -0.5 * torch.log(c) - torch.log(sigma) - 0.5 * ((x - mu) ** 2) / (sigma ** 2)



