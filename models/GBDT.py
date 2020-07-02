from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import time
from sklearn.metrics import mean_squared_error
import numpy as np

class GBDT:
    def __init__(self, task='regression', depth=8):
        self.task = task
        self.depth = depth


    def init_model(self, num_epochs, learning_rate, patience):
        catboost_model_obj = CatBoostRegressor if self.task == 'regression' else CatBoostClassifier
        self.catboost_loss_function = 'CrossEntropy' if self.task == 'classification' else 'RMSE'

        self.model = catboost_model_obj(iterations=num_epochs,
                                       depth=self.depth,
                                       learning_rate=learning_rate,
                                       loss_function=self.catboost_loss_function,
                                       # custom_metric=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                                       random_seed=0,
                                        early_stopping_rounds=patience,
                                       nan_mode='Min')

    def get_accuracies(self):

        accuracies = [self.model.evals_result_['learn'][self.catboost_loss_function]]
        if 'validation' in self.model.evals_result_:
            accuracies.append([self.model.evals_result_['validation'][self.catboost_loss_function]])
        elif 'validation_0' in self.model.evals_result_:
            accuracies.extend([self.model.evals_result_['validation_0'][self.catboost_loss_function],
                               self.model.evals_result_['validation_1'][self.catboost_loss_function]
                               ])
        return list(zip(*accuracies))

    def get_test_accuracy(self, accuracies):
        min_val_epoch = np.argmin([acc[1] for acc in accuracies])
        min_rmse = accuracies[min_val_epoch]
        return min_rmse, min_val_epoch

    def save_accuracies(self, accuracies, fn):
        with open(fn, "w+") as f:
            for acc in accuracies:
                print(*acc, file=f)

    def train_val_test_split(self, X, y, train_mask, val_mask, test_mask):
        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit(self,
            X, y, train_mask, val_mask, test_mask,
            cat_features=None, num_epochs=1000, patience=200,
            learning_rate=None, plot=True, verbose=False,
            loss_fn=""):

        X_train, y_train, X_val, y_val, X_test, y_test = \
            self.train_val_test_split(X, y, train_mask, val_mask, test_mask)
        self.init_model(num_epochs, learning_rate, patience)

        start = time.time()
        pool = Pool(X_train, y_train, cat_features=cat_features)
        eval_set = [(X_val, y_val), (X_test, y_test)]
        self.model.fit(pool, eval_set=eval_set, plot=plot, verbose=verbose)
        finish = time.time()

        num_trees = self.model.tree_count_
        print('Finished training. Total time: {:.2f} | Number of trees: {:d} | Time per tree: {:.2f}'.format(finish - start, num_trees, (time.time() - start )/num_trees))

        accuracies = self.get_accuracies()
        min_rmse, min_rmse_epoch = self.get_test_accuracy(accuracies)
        if loss_fn:
            self.save_accuracies(accuracies, loss_fn)
        print('Best iteration {} with accuracy {:.3f}/{:.3f}/{:.3f}'.format(min_rmse_epoch, *min_rmse))

    def predict(self, X_test, y_test):
        pred = self.model.predict(X_test)

        metrics = {}
        metrics['rmse'] = mean_squared_error(pred, y_test) ** .5

        return metrics