import itertools
import torch
from sklearn import preprocessing
import pandas as pd
import torch.nn.functional as F
import numpy as np

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def pandas_to_torch(self, *args):
        return [torch.from_numpy(arg.to_numpy()).float().squeeze().to(self.device) for arg in args]

    def networkx_to_torch(self, networkx_graph):
        import dgl
        graph = dgl.DGLGraph()
        graph.from_networkx(networkx_graph)
        graph = graph.to(self.device)
        return graph

    def move_to_device(self, *args):
        return [arg.to(self.device) for arg in args]

    def init_optimizer(self, node_features, optimize_node_features, learning_rate):

        params = [self.model.parameters()]
        if optimize_node_features:
            params.append([node_features])
        optimizer = torch.optim.AdamW(itertools.chain(*params), lr=learning_rate)
        return optimizer

    def log_epoch(self, pbar, accuracies, epoch, loss, epoch_time, logging_epochs):
        train_rmse, val_rmse, test_rmse = accuracies[-1]
        if epoch and epoch % logging_epochs == 0:
            pbar.set_description(
                "Epoch {:05d} | Loss {:.3f} | RMSE {:.3f}/{:.3f}/{:.3f} | Time {:.4f}".format(epoch, loss,
                                                                                              train_rmse,
                                                                                              val_rmse, test_rmse,
                                                                                              epoch_time))

    def normalize_features(self, X, train_mask, val_mask, test_mask):
        min_max_scaler = preprocessing.MinMaxScaler()
        A = X.to_numpy()
        A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
        A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
        return pd.DataFrame(A, columns=X.columns).astype(float)

    def encode_cat_features(self, X, y, cat_features, train_mask, val_mask, test_mask):
        from category_encoders import CatBoostEncoder
        enc = CatBoostEncoder()
        A = X.to_numpy()
        b = y.to_numpy()
        A[np.ix_(train_mask, cat_features)] = enc.fit_transform(A[np.ix_(train_mask, cat_features)], b[train_mask])
        A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(A[np.ix_(val_mask + test_mask, cat_features)])
        A = A.astype(float)
        return pd.DataFrame(A, columns=X.columns)

    def train_model(self, model_in, target_labels, train_mask, optimizer):
        self.model.train()
        y = target_labels[train_mask]
        logits = self.model(*model_in)[train_mask].squeeze()

        loss = torch.sqrt(F.mse_loss(logits, y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def evaluate_model(self, model_in, target_labels, mask):
        self.model.eval()
        metrics = {}
        y = target_labels[mask]
        with torch.no_grad():
            pred = self.model(*model_in).squeeze()[mask]
            metrics['rmse'] = torch.sqrt(F.mse_loss(pred, y).squeeze() + 1e-8)
            metrics['rmsle'] = torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(y + 1)).squeeze() + 1e-8)
            metrics['mae'] = F.l1_loss(pred, y)
            return metrics

    def train_val_test_split(self, X, y, train_mask, val_mask, test_mask):
        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_and_evaluate(self, model_in, target_labels, train_mask, val_mask, test_mask,
                           optimizer, accuracies, gnn_passes_per_epoch):
        loss = None
        for _ in range(gnn_passes_per_epoch):
            loss = self.train_model(model_in, target_labels, train_mask, optimizer)
        train_results = self.evaluate_model(model_in, target_labels, train_mask)
        val_results = self.evaluate_model(model_in, target_labels, val_mask)
        test_results = self.evaluate_model(model_in, target_labels, test_mask)
        accuracies.append((train_results['rmse'].detach().item(),
                           val_results['rmse'].detach().item(),
                           test_results['rmse'].detach().item()))
        return loss

    def update_early_stopping(self, accuracies, epoch, min_rmse, min_rmse_epoch, epochs_since_last_min_rmse):
        train_rmse, val_rmse, test_rmse = accuracies[-1]
        if val_rmse < min_rmse[1]:
            min_rmse = accuracies[-1]
            min_rmse_epoch = epoch
            epochs_since_last_min_rmse = 0
        else:
            epochs_since_last_min_rmse += 1
        return min_rmse, min_rmse_epoch, epochs_since_last_min_rmse

    def save_accuracies(self, accuracies, fn):
        with open(fn, "w+") as f:
            for acc in accuracies:
                print(*acc, file=f)

    def plot(self, accuracies, legend, title, output_fn=None, logx=False, logy=False):
        import matplotlib.pyplot as plt

        xs = [range(len(accuracies))] * len(accuracies[0])
        ys = list(zip(*accuracies))

        plt.rcParams.update({'font.size': 40})
        plt.rcParams["figure.figsize"] = (20, 10)
        lss = ['-', '--', '-.', ':']
        colors = ['#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']
        colors = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2),
                  (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
        colors = [[p / 255 for p in c] for c in colors]
        for i in range(len(ys)):
            plt.plot(xs[i], ys[i], lw=4, color=colors[i])
        plt.legend(legend, loc=1, fontsize=30)
        plt.title(title)

        plt.xscale('log') if logx else None
        plt.yscale('log') if logy else None
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.grid()
        plt.tight_layout()

        plt.savefig(output_fn, bbox_inches='tight') if output_fn else None
        plt.show()

    def plot_interactive(self, accuracies, legend, title, logx=False, logy=False):
        import plotly.graph_objects as go

        xs = [list(range(len(accuracies)))] * len(accuracies[0])
        ys = list(zip(*accuracies))

        fig = go.Figure()
        for i in range(len(ys)):
            fig.add_trace(go.Scatter(x=xs[i], y=ys[i],
                                     mode='lines+markers',
                                     name=legend[i]))

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title='Epoch',
            yaxis_title='RMSE',
            font=dict(
                size=40,
            ),
            height=600,
        )

        if logx:
            fig.update_layout(xaxis_type="log")
        if logy:
            fig.update_layout(yaxis_type="log")

        fig.show()