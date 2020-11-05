import time
import numpy as np
import torch
from torch.nn import Dropout, ELU
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv as GATConvDGL, GraphConv, ChebConv as ChebConvDGL, \
    AGNNConv as AGNNConvDGL, APPNPConv
from torch_geometric.nn import GATConv, SplineConv, GCNConv, ChebConv, GINConv, GraphUNet, AGNNConv
from torch.nn import Sequential, Linear, ReLU, Identity
from tqdm import tqdm
from .Base import BaseModel
from torch.autograd import Variable
from collections import defaultdict as ddict
from .MLP import MLPRegressor

class GNNModelDGL(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 heads=8, dropout=0., name='gat', residual=True, use_mlp=False, join_with_mlp=False):
        super(GNNModelDGL, self).__init__()
        self.name = name
        self.use_mlp = use_mlp
        self.join_with_mlp = join_with_mlp
        if use_mlp:
            self.mlp = MLPRegressor(in_dim, hidden_dim, out_dim)
            if join_with_mlp:
                in_dim += 1
            else:
                in_dim = out_dim
        if name == 'gat':
            self.l1 = GATConvDGL(in_dim, hidden_dim, heads, feat_drop=dropout, attn_drop=dropout, residual=False,
                              activation=F.elu)
            self.l2 = GATConvDGL(hidden_dim * heads, out_dim, 1, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=None)
        elif name == 'gcn':
            self.l1 = GraphConv(in_dim, hidden_dim, activation=F.elu)
            self.l2 = GraphConv(hidden_dim, out_dim, activation=F.elu)
            self.drop = Dropout(p=dropout)
        elif name == 'cheb':
            self.l1 = ChebConvDGL(in_dim, hidden_dim, k = 3)
            self.l2 = ChebConvDGL(hidden_dim, out_dim, k = 3)
            self.drop = Dropout(p=dropout)
        elif name == 'agnn':
            self.lin1 = Sequential(Dropout(p=dropout), Linear(in_dim, hidden_dim), ELU())
            self.l1 = AGNNConvDGL(learn_beta=False)
            self.l2 = AGNNConvDGL(learn_beta=True)
            self.lin2 = Sequential(Dropout(p=dropout), Linear(hidden_dim, out_dim), ELU())
        elif name == 'appnp':
            self.lin1 = Sequential(Dropout(p=dropout), Linear(in_dim, hidden_dim),
                       ELU(), Dropout(p=dropout), Linear(hidden_dim, out_dim), ELU())
            self.l1 = APPNPConv(k=10, alpha=0.1)


    def forward(self, graph, features):
        h = features
        if self.use_mlp:
            if self.join_with_mlp:
                h = torch.cat((h, self.mlp(features)), 1)
            else:
                h = self.mlp(features)
        if self.name == 'gat':
            h = self.l1(graph, h).flatten(1)
            logits = self.l2(graph, h).mean(1)
        elif self.name in ['appnp']:
            h = self.lin1(h)
            logits = self.l1(graph, h)
        elif self.name == 'agnn':
            h = self.lin1(h)
            h = self.l1(graph, h)
            h = self.l2(graph, h)
            logits = self.lin2(h)
        elif self.name in ['gcn', 'cheb']:
            h = self.drop(h)
            h = self.l1(graph, h)
            logits = self.l2(graph, h)


        return logits

class GNNModelPYG(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 dropout=0.5, name='gat',
                 heads=8, residual=True):
        super(GNNModelPYG, self).__init__()
        self.dropout = dropout
        self.name = name
        self.residual = None
        if residual:
            if in_dim == out_dim:
                self.residual = Identity()
            else:
                self.residual = Linear(in_dim, out_dim)

        if name == 'gat':
            self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        elif name == 'gcn':
            self.conv1 = GCNConv(in_dim, hidden_dim, cached=True, normalize=True, add_self_loops=False)
            self.conv2 = GCNConv(hidden_dim, out_dim, cached=True, normalize=True, add_self_loops=False)
        elif name == 'cheb':
            self.conv1 = ChebConv(in_dim, hidden_dim, K=2)
            self.conv2 = ChebConv(hidden_dim, out_dim, K=2)
        elif name == 'spline':
            self.conv1 = SplineConv(in_dim, hidden_dim, dim=1, kernel_size=2)
            self.conv2 = SplineConv(hidden_dim, out_dim, dim=1, kernel_size=2)
        elif name == 'gin':
            self.conv1 = GINConv(Sequential(Linear(in_dim, hidden_dim),
                                            ReLU(), Linear(hidden_dim, hidden_dim)))
            self.conv2 = GINConv(Sequential(Linear(hidden_dim, hidden_dim),
                                            ReLU(), Linear(hidden_dim, out_dim)))
        elif name == 'unet':
            self.conv1 = GraphUNet(in_dim, hidden_dim, out_dim, depth=3)
        elif name == 'agnn':
            self.lin1 = Linear(in_dim, hidden_dim)
            self.conv1 = AGNNConv(requires_grad=False)
            self.conv2 = AGNNConv(requires_grad=True)
            self.lin2 = Linear(hidden_dim, out_dim)
        else:
            raise NotImplemented("""
            Unknown model name. Choose from gat, gcn, cheb, spline, gin, unet, agnn.""")

    def forward(self, graph, features):
        x, edge_index, edge_attr = features, graph.edge_index, graph.edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)
        model_in = [x, edge_index]
        if self.name in ['spline']: # use edge attributes
            model_in.append(edge_attr)

        if self.name in ['unet']: # only one layer (net)
            x = self.conv1(*model_in)
        elif self.name in ['agnn']:
            x = F.elu(self.conv1(self.lin1(x), edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(self.conv2(x, edge_index))
        else:
            x = F.elu(self.conv1(*model_in))
            x = F.dropout(x, p=self.dropout, training=self.training)
            model_in[0] = x
            x = self.conv2(*model_in)
        if self.residual is not None:
            x += self.residual(features)
        return x


class GNN(BaseModel):
    def __init__(self, heads=8, dropout=0.5, task='regression', name='gat', residual=True, lang='dgl',
                 use_mlp = False, join_with_mlp=False):
        super(GNN, self).__init__()

        self.heads = heads
        self.dropout = dropout
        self.task = task
        self.model_name = name
        self.use_residual = residual
        self.lang = lang
        self.use_mlp = use_mlp
        self.join_with_mlp = join_with_mlp

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def init_model(self):
        if self.lang == 'pyg':
            self.model = GNNModelPYG(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                                  heads=self.heads, dropout=self.dropout, name=self.model_name,
                                  residual=self.use_residual).to(self.device)
        elif self.lang == 'dgl':
            self.model = GNNModelDGL(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                                     heads=self.heads, dropout=self.dropout, name=self.model_name,
                                     residual=self.use_residual, use_mlp=self.use_mlp,
                                     join_with_mlp=self.join_with_mlp).to(self.device)

    def init_node_features(self, X, optimize_node_features):
        node_features = Variable(X, requires_grad=optimize_node_features)
        return node_features

    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, num_epochs,
            cat_features=None, optimize_node_features=False, learning_rate=1e-2, patience=200, logging_epochs=1,
            hidden_dim=8, gbdt_predictions=None, only_gbdt=False,
            loss_fn=None, metric_name='loss', normalize_features=True, replace_na=True):

        # initialize for early stopping and metrics
        min_metric = [np.float('inf')] * 3  # for train/val/test
        min_val_epoch = 0
        epochs_since_last_min_metric = 0
        metrics = ddict(list) # metric_name -> (train/val/test)
        if cat_features is None:
            cat_features = []

        if gbdt_predictions is not None:
            X = X.copy()
            X['predict'] = gbdt_predictions
            if only_gbdt:
                cat_features = []
                X = X[['predict']]

        self.in_dim = X.shape[1]
        self.hidden_dim = hidden_dim
        if self.task == 'regression':
            self.out_dim = y.shape[1]
        elif self.task == 'classification':
            self.out_dim = len(set(y.iloc[:, 0]))

        if len(cat_features):
            X = self.encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask)
        if normalize_features:
            X = self.normalize_features(X, train_mask, val_mask, test_mask)
        if replace_na:
            X = self.replace_na(X, train_mask)

        X, y = self.pandas_to_torch(X, y)
        if len(X.shape) == 1:
            X = X.unsqueeze(1)

        if self.lang == 'dgl':
            graph = self.networkx_to_torch(networkx_graph)
        elif self.lang == 'pyg':
            graph = self.networkx_to_torch2(networkx_graph)

        self.init_model()
        node_features = self.init_node_features(X, optimize_node_features)

        self.node_features = node_features
        self.graph = graph
        optimizer = self.init_optimizer(node_features, optimize_node_features, learning_rate)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()

            model_in = (graph, node_features)
            loss = self.train_and_evaluate(model_in, y, train_mask, val_mask, test_mask, optimizer,
                                           metrics, gnn_passes_per_epoch=1)
            self.log_epoch(pbar, metrics, epoch, loss, time.time() - start2epoch, logging_epochs)

            # check early stopping
            min_metric, min_val_epoch, epochs_since_last_min_metric = \
                self.update_early_stopping(metrics, epoch, min_metric, min_val_epoch, epochs_since_last_min_metric,
                                           metric_name)
            if epochs_since_last_min_metric > patience:
                break

        if loss_fn:
            self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, min_val_epoch, *min_metric))
        return min_val_epoch, metrics

    def predict(self, graph, node_features, target_labels, test_mask):
        return self.evaluate_model((graph, node_features), target_labels, test_mask)