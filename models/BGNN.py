import time
import numpy as np
import torch

from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from .GNN import GATModel
from .Base import BaseModel
from tqdm import tqdm

class BGNN(BaseModel):
    def __init__(self,
                 task='regression', depth=8, heads=8, feat_drop=0, attn_drop=0, resgnn=False, train_residual=False):
        super(BaseModel, self).__init__()
        self.task = task
        self.depth = depth
        self.heads = heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.resgnn = resgnn
        self.train_residual = train_residual

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def init_gbdt_model(self, num_epochs):
        catboost_model_obj = CatBoostRegressor if self.task == 'regression' else CatBoostClassifier
        self.catboost_loss_function = 'CrossEntropy' if self.task == 'classification' else 'RMSE'
        return catboost_model_obj(iterations=num_epochs,
                                  depth=self.depth,
                                  learning_rate=0.1,
                                  loss_function=self.catboost_loss_function,
                                  random_seed=0,
                                  nan_mode='Min')

    def fit_gbdt(self, pool, trees_per_epoch):
        gbdt_model = self.init_gbdt_model(trees_per_epoch)
        gbdt_model.fit(pool, verbose=False)
        return gbdt_model

    def init_gnn_model(self):
        self.model = GATModel(in_dim=self.in_dim,
                              hidden_dim=self.hidden_dim,
                              out_dim=self.out_dim,
                              heads=self.heads,
                              feat_drop=self.feat_drop,
                              attn_drop=self.attn_drop).to(self.device)

    def append_gbdt_model(self, new_gbdt_model, weights):
        if self.gbdt_model is None:
            return new_gbdt_model
        return sum_models([self.gbdt_model, new_gbdt_model], weights=weights)

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, cat_features,
                   gbdt_trees_per_epoch, gbdt_alpha):

        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features)
        epoch_gbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch)
        self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, gbdt_alpha])

    def update_node_features(self, node_features, X, encoded_X):
        predictions = self.gbdt_model.predict(X)

        if self.resgnn:
            if self.train_residual:
                predictions = np.append(node_features.detach().cpu().data[:, :-1], np.expand_dims(predictions, axis=1),
                                        axis=1)  # append updated X to prediction
            else:
                predictions = np.append(encoded_X, np.expand_dims(predictions, axis=1), axis=1)  # append X to prediction

        predictions = torch.from_numpy(predictions).to(self.device)
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(1)

        node_features.data = predictions.float().data

    def update_gbdt_targets(self, node_features, node_features_before, train_mask):
        column = -1 if self.resgnn else 0
        return (node_features - node_features_before).detach().cpu().numpy()[train_mask, column]

    def init_node_features(self, X):
        node_features = torch.empty(X.shape[0], self.in_dim, requires_grad=True, device=self.device)
        if self.resgnn:
            node_features.data[:, :-1] = torch.from_numpy(X.to_numpy())
        return node_features

    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, num_epochs, patience=50,
            hidden_dim=8, gbdt_trees_per_epoch=1, gnn_passes_per_epoch=1, cat_features=None, learning_rate=1e-2,
            logging_epochs=1, loss_fn=None
            ):

        # initialize for early stopping and accuracies
        min_rmse = [np.float('inf')] * 3  # for train/val/test
        min_rmse_epoch = 0
        epochs_since_last_min_rmse = 0
        accuracies = []
        grad_norm = []
        if cat_features is None:
            cat_features = []

        self.out_dim = y.shape[1]
        self.in_dim = self.out_dim + X.shape[1] if self.resgnn else self.out_dim
        self.hidden_dim = hidden_dim
        self.init_gnn_model()

        gbdt_X_train = X.iloc[train_mask]
        gbdt_y_train = y.iloc[train_mask]
        gbdt_alpha = 1
        self.gbdt_model = None

        encoded_X = X.copy()
        if self.resgnn:
            if len(cat_features):
                encoded_X = self.encode_cat_features(encoded_X, y, cat_features, train_mask, val_mask, test_mask)
            encoded_X = self.normalize_features(encoded_X, train_mask, val_mask, test_mask)

        node_features = self.init_node_features(encoded_X)
        optimizer = self.init_optimizer(node_features, optimize_node_features=True, learning_rate=learning_rate)

        y, = self.pandas_to_torch(y)
        graph = self.networkx_to_torch(networkx_graph)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()

            # gbdt part
            self.train_gbdt(gbdt_X_train, gbdt_y_train, cat_features,
                            gbdt_trees_per_epoch, gbdt_alpha)
            self.update_node_features(node_features, X, encoded_X)

            # gnn part
            node_features_before = node_features.clone()

            model_in=(graph, node_features)
            loss = self.train_and_evaluate(model_in, y, train_mask, val_mask, test_mask,
                                           optimizer, accuracies, gnn_passes_per_epoch)
            gbdt_y_train = self.update_gbdt_targets(node_features, node_features_before, train_mask)
            grad_norm.append(np.linalg.norm(gbdt_y_train))

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

    def predict(self, graph, X, y, test_mask):
        node_features = torch.empty(X.shape[0], self.in_dim).to(self.device)
        self.update_node_features(node_features, X)
        return self.evaluate_model((graph, node_features), y, test_mask)