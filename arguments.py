import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--missing_link', type=float, default=0.0)
    parser.add_argument('--missing_feature', type=float, default=0.0)
    parser.add_argument('--train_per_class', type=int, default=23)
    parser.add_argument('--val_per_class', type=int, default=30)
    parser.add_argument('--ogb_train_ratio', type=float, default=1.0)

    parser.add_argument('--num_trials', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--normalize_features', type=bool, default=True)

    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=256)

    parser.add_argument('--lambda_pa', type=float, default=4)
    parser.add_argument('--lambda_ce_aug', type=float, default=0.2)
    parser.add_argument('--num_neighbor', type=int, default=20)
    parser.add_argument('--knn_metric', type=str, default='cosine', choices=['cosine','minkowski'])
    parser.add_argument('--batch_size', type=int, default=0)

    # missing_link: 假设是用于模拟图中缺失边的比例。
    # missing_feature: 用于模拟图中节点特征缺失的比例。
    # train_per_class: 每个类别用于训练的节点数。
    # val_per_class: 每个类别用于验证的节点数。
    # ogb_train_ratio: 对于OGB数据集，用于定义训练集比例。
    # num_trials: 进行试验的次数，用于平均模型性能评估。
    # dataset: 使用的数据集名称，如'Cora'、'Citeseer'等。
    # dropout: 在模型中使用的dropout比率，防止过拟合。
    # epochs: 训练的最大轮数。
    # lr: 学习率，影响模型训练过程中权重调整的速度。
    # weight_decay: 权重衰减，用于正则化。
    # patience: 早停的耐心值，如果验证损失在设定的轮数内未改善，则停止训练。
    # use_bn: 是否使用批量归一化。
    # normalize_features: 是否标准化输入特征。
    # T: 特征传播的步数。
    # alpha: 特征传播过程中的衰减参数。
    # hidden: 隐藏层的大小。
    # lambda_pa: 原型对齐损失的权重。
    # lambda_ce_aug: 数据增强交叉熵损失的权重。
    # num_neighbor: 在KNN图中每个节点的邻居数。
    # knn_metric: 用于KNN图的距离度量。
    # batch_size: 训练时的批大小，0表示全批量处理。
    return parser.parse_args()