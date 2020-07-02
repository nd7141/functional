import time
import numpy as np
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from tqdm import tqdm
from .Base import BaseModel
from torch.autograd import Variable


class GATModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 heads,
                 activation=F.elu,
                 feat_drop=0,
                 attn_drop=0,
                 residual=True):
        super(GATModel, self).__init__()
        self.l1 = GATConv(in_dim, hidden_dim, heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=False,
                          activation=activation)
        self.l2 = GATConv(hidden_dim * heads, out_dim, 1, residual=residual, activation=None)

    def forward(self, graph, features):
        h = features
        h = self.l1(graph, h).flatten(1)
        logits = self.l2(graph, h).mean(1)
        return logits


class GNN(BaseModel):
    def __init__(self, heads=8, feat_drop=0, attn_drop=0):
        super(GNN, self).__init__()

        self.heads = heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def init_model(self):
        self.model = GATModel(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                              heads=self.heads, feat_drop=self.feat_drop, attn_drop=self.attn_drop).to(self.device)

    def init_node_features(self, X, optimize_node_features):
        node_features = node_features = Variable(X, requires_grad=optimize_node_features)
        return node_features

    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, num_epochs,
            cat_features=None, optimize_node_features=False, learning_rate=1e-2, patience=200, logging_epochs=1,
            hidden_dim=8,
            loss_fn=None):

        # initialize for early stopping and accuracies
        min_rmse = [np.float('inf')] * 3  # for train/val/test
        min_rmse_epoch = 0
        epochs_since_last_min_rmse = 0
        accuracies = []
        if cat_features is None:
            cat_features = []

        self.in_dim = X.shape[1]
        self.hidden_dim = hidden_dim
        self.out_dim = y.shape[1]

        X = self.normalize_features(X, train_mask, val_mask, test_mask)
        if len(cat_features):
            X = self.encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask)

        X, y = self.pandas_to_torch(X, y)
        graph = self.networkx_to_torch(networkx_graph)

        self.init_model()
        node_features = self.init_node_features(X, optimize_node_features)
        optimizer = self.init_optimizer(node_features, optimize_node_features, learning_rate)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()

            model_in = (graph, node_features)
            loss = self.train_and_evaluate(model_in, y, train_mask, val_mask, test_mask, optimizer,
                                           accuracies, gnn_passes_per_epoch=1)
            self.log_epoch(pbar, accuracies, epoch, loss, time.time() - start2epoch, logging_epochs)

            # check early stopping
            min_rmse, min_rmse_epoch, epochs_since_last_min_rmse = \
                self.update_early_stopping(accuracies, epoch, min_rmse, min_rmse_epoch, epochs_since_last_min_rmse)
            if patience and epochs_since_last_min_rmse > patience:
                break

        if loss_fn:
            self.save_accuracies(accuracies, loss_fn)
        print('Best iteration {} with accuracy {:.3f}/{:.3f}/{:.3f}'.format(min_rmse_epoch, *min_rmse))
        return min_rmse_epoch, accuracies

    def predict(self, graph, node_features, target_labels, test_mask):
        return self.evaluate_model((graph, node_features), target_labels, test_mask)