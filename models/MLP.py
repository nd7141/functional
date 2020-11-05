import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from .Base import BaseModel
from sklearn.metrics import r2_score
from collections import defaultdict as ddict

class MLPClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super(MLPClassifier, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_dim, hidden_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.lins.append(torch.nn.Linear(hidden_dim, out_dim))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super(MLPRegressor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class MLP(BaseModel):
    def __init__(self, task='regression'):
        super(MLP, self).__init__()
        self.task = task

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def init_model(self):
        mlp_model = MLPRegressor if self.task == 'regression' else MLPClassifier
        self.model = mlp_model(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim).to(
            self.device)

    def fit(self, X, y, train_mask, val_mask, test_mask, cat_features=None, num_epochs=1000,
            learning_rate=1e-2, patience=200, hidden_dim=128, logging_epochs=1, loss_fn=None,
            metric_name='loss', normalize_features=True, replace_na=True):

        # initialize for early stopping and metrics
        min_metric = [np.float('inf')] * 3  # for train/val/test
        min_val_epoch = 0
        epochs_since_last_min_metric = 0
        metrics = ddict(list) # metric_name -> (train/val/test)
        if cat_features is None:
            cat_features = []

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

        self.init_model()
        optimizer = self.init_optimizer(node_features=None, optimize_node_features=False,
                                        learning_rate=learning_rate)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:

            start2epoch = time.time()

            model_in = (X,)
            loss = self.train_and_evaluate(model_in, y, train_mask, val_mask, test_mask, optimizer,
                                           metrics, gnn_passes_per_epoch=1)
            self.log_epoch(pbar, metrics, epoch, loss, time.time() - start2epoch, logging_epochs)

            # check early stopping
            min_metric, min_val_epoch, epochs_since_last_min_metric = \
                self.update_early_stopping(metrics, epoch, min_metric, min_val_epoch, epochs_since_last_min_metric, metric_name)
            if epochs_since_last_min_metric > patience:
                break

        if loss_fn:
            self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, min_val_epoch, *min_metric))
        return min_val_epoch, metrics

    def predict(self, X, target_labels, test_mask):
        return self.evaluate_model((X,), target_labels, test_mask)