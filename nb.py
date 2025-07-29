class BayesianLinear(torch.nn.Module):
    """
    Bayesian linear layer with normal prior and normal posterior
    """

    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        #神经网络中线性层（全连接层）的构造函数中的两个参数，用于定义该层的输入和输出维度。其中，in_features 表示该层的输入维度，即输入数据的特征数；
        #out_features 表示该层的输出维度，即该层的神经元数量或输出数据的特征数。这两个参数在初始化该层的权重矩阵和偏置向量时会用到。
        #具体来说，权重矩阵的形状为 (out_features, in_features)，即该层的每个神经元与输入数据的每个特征都有一个对应的权重；
        #偏置向量的形状为 (out_features,)，即该层的每个神经元都有一个对应的偏置项。这两个参数的值通常需要根据具体的任务和数据来进行调整，以获得更好的性能。
        self.in_features = in_features
        self.out_features = out_features

        #weight_mu 和 weight_rho 分别表示权重的均值和标准差，它们都是可学习参数，需要在训练过程中不断地更新。
        #weight_mu 是一个形状为 (out_features, in_features) 的张量，表示权重的均值
        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        #weight_rho 是一个形状也为 (out_features, in_features) 的张量，表示权重的标准差
        #因为标准差需要保持在一个较小的范围内，以便在训练过程中能够更好地学习权重的分布。
        self.weight_rho = torch.nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        # bias_mu 和 bias_rho 分别表示偏置的均值和标准差，它们都是可学习参数，需要在训练过程中不断地更新。
        #bias_mu 是一个形状为 (out_features,) 的张量，表示偏置的均值
        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        #bias_rho 是一个形状也为 (out_features,) 的张量，表示偏置的标准差
        self.bias_rho = torch.nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        #log_prior 和 log_variational_posterior 分别表示该层的先验分布和变分后验分布的对数概率密度函数（log-pdf）
        #log_prior 表示该层的先验分布的对数概率密度函数。在这里，我们可以将其设置为常数0.0，表示先验分布为标准正态分布，即权重和偏置在训练前没有任何先验信息。
        self.log_prior = 0.0
        #log_variational_posterior 表示该层的变分后验分布的对数概率密度函数。
        self.log_variational_posterior = 0.0
        #在训练过程中，我们需要通过最小化 KL 散度来学习变分后验分布，使其逐渐逼近真实的后验分布。具体计算 KL 散度的公式为 KL(q(w)||p(w)) = log(q(w)/p(w)) + E_q[log(p(D|w))] - log(p(D))，
        #其中 q(w) 表示变分后验分布，p(w) 表示先验分布，p(D|w) 表示数据在给定权重和偏置下的概率密度函数，E_q[log(p(D|w))] 表示数据的对数似然，log(p(D)) 表示数据的边缘概率密度函数。
        #我们可以通过对 log(q(w)/p(w)) 和 E_q[log(p(D|w))] 的梯度进行反向传播来更新变分后验分布的参数。


    def forward(self, x):
        #weight_epsilon 和 bias_epsilon 分别表示权重和偏置的采样噪声，它们是形状为 (out_features, in_features) 和 (out_features,) 的张量。randn正态分布函数
        weight_epsilon = torch.randn(self.out_features, self.in_features).to(x.device)
        bias_epsilon = torch.randn(self.out_features).to(x.device)
        #weight_std 和 bias_std 分别表示权重和偏置分布的对数标准差，它们是形状为 (out_features, in_features) 和 (out_features,)
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        #weight 和 bias 表示使用采样噪声计算得到的具体的权重和偏置值。
        weight = self.weight_mu + weight_std * weight_epsilon
        bias = self.bias_mu + bias_std * bias_epsilon
        #self.log_prior 和 self.log_variational_posterior 分别表示模型的先验概率和后验概率的对数值。
        #elf.log_prior 表示模型参数的先验概率的负对数和的对数值，它由权重和偏置的先验概率密度函数计算得到，这里假设先验分布为正态分布，均值为0，标准差为1。
        self.log_prior = normal_logpdf(weight, 0, 1).sum() + normal_logpdf(bias, 0, 1).sum()
        #self.log_variational_posterior 表示模型参数的后验概率的负对数和的对数值，它由权重和偏置的后验概率密度函数计算得到，这里假设后验分布为正态分布，
        self.log_variational_posterior = normal_logpdf(weight, self.weight_mu, weight_std).sum() + normal_logpdf(bias,
                                                                                                                 self.bias_mu,
                                                                                                                 bias_std).sum()
        return F.linear(x, weight, bias)


#normal_logpdf 函数是用于计算给定均值 mu 和标准差 sigma 的正态分布在 x 处的对数概率密度值。
def normal_logpdf(x, mu, sigma):
    """
    Computes the log pdf of a normal distribution with mean mu and variance sigma at x
    """
    a = np.array(2 * torch.tensor([math.pi]))
    b = torch.from_numpy(a)
    c = torch.Tensor(b)
    return -0.5 * torch.log(c) - torch.log(torch.Tensor(sigma)) - 0.5 * ((x - mu) ** 2) / (sigma ** 2)
#其中，log 表示以自然对数为底的对数函数，pi 表示圆周率，sigma 表示标准差。在这里，我们使用 torch 库来计算对数概率密度值。
#具体来说，-0.5 * torch.log(c) 表示计算 -0.5 * log(2 * pi) 的值，torch.log(torch.Tensor(sigma)) 表示计算 log(sigma) 的值，
#- 0.5 * ((x - mu) ** 2) / (sigma ** 2) 表示计算 - 0.5 * ((x - mu) ** 2) / (sigma ** 2) 的值。最终返回的是正态分布在 x 处的对数概率密度值。

class GCNNodeCLF(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GCNNodeCLF, self).__init__()
        self.num_layers = nlayer#模型层数
        self.dropout = dropout#练过程中用于正则化的dropout率
        self.nclass = nclass#分类任务中的类别数
        self.prior_mu = 0#先验分布的均值
        self.prior_sigma = 1#先验分布的标准差

        #始化了两个空列表 self.posterior_mu 和 self.posterior_sigma，分别用 torch.nn.ParameterList() 类型来存储模型每层的后验均值和标准差参数。
        # 在训练过程中，这些参数会通过反向传播来学习。
        self.posterior_mu = torch.nn.ParameterList()
        self.posterior_sigma = torch.nn.ParameterList()

        #其中第一层为贝叶斯线性层（BayesianLinear），输入特征数为 nfeat，输出特征数为 nhid
        self.pre = torch.nn.Sequential(BayesianLinear(nfeat, nhid), torch.nn.ReLU())

        self.graph_convs = torch.nn.ModuleList()
        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))

        self.post = BayesianLinear(nhid, nclass)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.post(x)
        return F.log_softmax(x, dim=1)

    #模型的损失函数由两部分组成，即负对数似然损失和 KL 散度损失。
    # 其中，负对数似然损失衡量了模型预测结果和真实结果之间的差异，KL 散度损失衡量了后验分布和先验分布之间的差异，是一个正则化项，可以约束后验分布的分布范围，防止过拟合。
    def loss(self, pred, label):
        nll_loss = F.nll_loss(pred,label)
        #分类模型过度学习了训练实例，导致模型在测试时效用低。可以通过在损失函数中加入正则化罚项来避免模型过拟合
        label = F.one_hot(label,self.nclass).to(torch.float32)
        kl_loss = F.kl_div(pred,label,reduction='batchmean')
        return nll_loss + kl_loss