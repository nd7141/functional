import argparse
from utils import get_masks
import pandas as pd
import networkx as nx
import os

#TODO: fix boolean arguments
#TODO: test it
#TODO: create tests on house dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running model on dataset')
    parser.add_argument('--model', choices=['gbdt', 'mlp', 'gnn', 'bgnn'], help='Model to run')
    parser.add_argument('--dataset_folder', default='', help='Folder with X.csv, y.csv, graph.graphml')
    parser.add_argument('--X', default='./data/house/X.csv', help='Pandas file with node features X')
    parser.add_argument('--y', default='./data/house/y.csv', help='Pandas file with target labels y')
    parser.add_argument('--graph', default='./data/house/X.csv', help='Graphml file with graph')
    parser.add_argument('--cat_features', nargs='+', type=int, help='List of categorical features (int columns)')

    parser.add_argument('--train_size', default=0.6, help='Train size', type=float)
    parser.add_argument('--val_size', default=0.2, help='Validation size', type=float)
    parser.add_argument('--random_seed', default=42, help='Random seed', type=int)

    parser.add_argument('--num_epochs', default=1000, help='Number of epochs to run', type=int)
    parser.add_argument('--patience', default=0, help='Waiting for early stopping (0 is no ES)', type=int)
    parser.add_argument('--lr', default=1e-2, help='Learning rate', type=float)
    parser.add_argument('--loss_fn', help='Text file to save accuracies', type=float)
    parser.add_argument('--hidden_dim', default=128, help='Hidden dimension', type=int)
    parser.add_argument('--task', default='regression', help='Regression or classification', type=str)
    parser.add_argument('--logging_epochs', default=1, help='Wait epochs to log', type=int)
    parser.add_argument('--input_grad', default=False, help='Optimize node features', type=bool)

    parser.add_argument('--heads', default=8, help='Number of heads', type=int)
    parser.add_argument('--feat_drop', default=0., help='GAT feat dropout rate', type=float)
    parser.add_argument('--attn_drop', default=0., help='GAT attn dropout rate', type=float)
    parser.add_argument('--resgnn', help='Append X to prediction', type=bool)
    parser.add_argument('--train_residual', help='Train residual features', type=bool)

    parser.add_argument('--trees_per_epoch', default=1, help='Number of trees per epoch', type=int)
    parser.add_argument('--backwards', default=1, help='Number of GNN backpropagation steps per epoch', type=int)

    args = parser.parse_args()

    if args.dataset_folder:
        X = pd.read_csv(args.dataset_folder + '/X.csv')
        y = pd.read_csv(args.dataset_folder + '/y.csv')
        networkx_graph = nx.read_graphml(args.dataset_folder + '/graph.graphml')
    else:
        X = pd.read_csv(args.X)
        y = pd.read_csv(args.y)
        networkx_graph = nx.read_graphml(args.graph)

    networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
    cat_features = args.cat_features

    train_mask, val_mask, train_val_mask, test_mask = get_masks(X.shape[0],
                                                                train_size=args.train_size,
                                                                val_size=args.val_size,
                                                                random_seed=args.random_seed, )

    os.makedirs('losses/', exist_ok=True)

    if args.model.lower() == 'gbdt':
        from models.GBDT import GBDT
        model = GBDT(depth=args.depth)
        model.fit(X, y, train_mask, val_mask, test_mask,
                 cat_features=cat_features, num_epochs=args.num_features, patience=args.patience,
                 learning_rate=args.learning_rate, plot=False, verbose=False,
                 loss_fn=args.loss_fn)

    elif args.model.lower() == 'mlp':
        from models.MLP import MLP
        model = MLP(task=args.task)
        min_rmse_epoch, accuracies = model.fit(X, y, train_mask, val_mask, test_mask,
                                               cat_features=cat_features, num_epochs=args.num_epochs, patience=args.patience,
                                               learning_rate=args.learning_rate, hidden_dim=args.hidden_dim,
                                               logging_epochs=args.logging_steps, loss_fn=args.loss_fn)

        model.plot(accuracies, legend=['Train', 'Val', 'Test'], title='MLP RMSE', output_fn='mlp_losses.pdf')
    elif args.model.lower() == 'gnn':
        from models.GNN import GNN
        model = GNN(heads=args.heads, feat_drop=args.feat_drop, attn_drop=args.attn_drop)

        min_rmse_epoch, accuracies = model.fit(networkx_graph, X, y, train_mask, val_mask, test_mask,
                                               cat_features=cat_features, num_epochs=args.num_epochs, patience=args.patience,
                                               learning_rate=args.learning_rate, hidden_dim=args.hidden_dim, logging_epochs=args.logging_steps,
                                               optimize_node_features=args.input_grad, loss_fn=args.loss_fn)

        model.plot(accuracies, legend=['Train', 'Val', 'Test'], title='GNN RMSE', output_fn='gnn_losses.pdf')
    elif args.model.lower() == 'bgnn':
        from models.BGNN import BGNN

        model = BGNN(task=args.task, heads=args.heads, feat_drop=args.feat_drop, attn_drop=args.attn_drop,
                     resgnn=args.resgnn, train_residual=args.train_residual)

        min_rmse_epoch, accuracies = model.fit(networkx_graph, X, y, train_mask, val_mask, test_mask,
                                               cat_features=cat_features, num_epochs=args.num_epochs,
                                               patience=args.patience,
                                               learning_rate=args.learning_rate, hidden_dim=args.hidden_dim,
                                               logging_epochs=args.logging_steps,
                                               gbdt_trees_per_epoch=args.trees_per_epoch, gnn_passes_per_epoch=args.backwards,
                                               loss_fn=args.loss_fn,)

        model.plot(accuracies, legend=['Train', 'Val', 'Test'], title='BGNN RMSE', output_fn='bgnn_losses.pdf')

