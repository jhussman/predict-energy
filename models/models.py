from torch import Tensor, save, load
from torch.nn import Linear, MSELoss, LeakyReLU, BatchNorm1d, Module, CrossEntropyLoss
from torch.optim import Adam
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np
import time
import pickle

sys.path.insert(0, "/Users/jhussman/Documents/predict-energy")
from utils import min_cross_entropy


"""
File containing model classes
-----------------------------


EnsembleModel
-A sci-kit learn based model that is defined as:
    -Ensemble-based model (i.e. GradientBoosting or RandomForest
    -Supports MSE or r2 metrics for regression and accuracy or AUC for classification

FeedForwardNN
-A Pytorch based model that is defined as:
    -3 Linear layers
    -2 Batch Normalization layers
    -Xavier uniform for initialization of weights
    -MSELoss for regression and CrossEntropy for classification
    -Adam optimizer
    -Supports MSE or r2 metrics for regression and accuracy for classification
    -LeakyReLU for all activation functions

Both model classes support regression or classification models

"""


class EnsembleModel:
    def __init__(self):

        # Define scalar and pipeline
        self.scalar = MinMaxScaler()
        self.pipeline = None

    def fit_scalar(self, train_features):
        """
        Fits scalar attribute to train_features parameter
        :param train_features: pandas DataFrame object or array, matrix of training features
        :return: Null
        """
        self.scalar.fit(train_features)

    def random_search(self, train_features, train_target, param_grid, model, k=5, metric='neg_mean_squared_error',
                      n_runs=10):
        """
        Performs Random Search Cross-validation to optimize specified model hyper-parameters

        :param train_features: pandas DataFrame object or array, matrix of training features
        :param train_target: pandas DataFrame object or array, vector of training target (or matrix if classification)
        :param param_grid: dictionary, parameters to explore in random search, follow sci-kit learn format
        :param model: object, sci-kit learn model object to explore (e.g. GradientBoostingClassifier)
        :param k: positive int, default 5, number of cross-validation folds
        :param metric: str, sci-kit learn metric to optimize hyper-parameters
        :param n_runs: positive int, default 10, number of hyper-parameter combos to explore
        :return: best parameters and best score
        """
        # Define random search object and explore
        grid = RandomizedSearchCV(Pipeline([('scalar', self.scalar), ('ensemble', model)]),
                                  param_grid, cv=k, scoring=metric, refit=True, verbose=2, n_jobs=-1, n_iter=n_runs)
        grid.fit(train_features, train_target)

        # Assign optimal model to pipeline attribute
        self.pipeline = Pipeline([('scalar', self.scalar), ('ensemble', grid.best_estimator_)])
        return grid.best_params_, grid.best_score_

    def grid_search(self, train_features, train_target, param_grid, model, k=5, metric='neg_mean_squared_error'):
        """
        Performs Grid Search Cross-validation to optimize specified model hyper-parameters

        :param train_features: pandas DataFrame object or array, matrix of training features
        :param train_target: pandas DataFrame object or array, vector of training target (or matrix if classification)
        :param param_grid: dictionary, parameters to explore in random search, follow sci-kit learn format
        :param model: object, sci-kit learn model object to explore (e.g. GradientBoostingClassifier)
        :param k: positive int, default 5, number of cross-validation folds
        :param metric: str, sci-kit learn metric to optimize hyper-parameters
        :return: best parameters and best score
        """
        # Define random search object and explore
        grid = GridSearchCV(Pipeline([('scalar', self.scalar), ('ensemble', model)]), param_grid, cv=k,
                            scoring=metric, refit=True, verbose=2, n_jobs=-1)
        grid.fit(train_features, train_target)

        # Assign optimal model to pipeline attribute
        self.pipeline = Pipeline([('scalar', self.scalar), ('ensemble', grid.best_estimator_)])
        return grid.best_params_, grid.best_score_

    def cross_validate(self, train_features, train_target, model, k=5, metric='neg_mean_squared_error'):
        """
        Performs cross-validation on a single instance of a sci-kit learn model object

        :param train_features: pandas DataFrame object or array, matrix of training features
        :param train_target: pandas DataFrame object or array, vector of training target (or matrix if classification)
        :param model: object, sci-kit learn model object to cross-validate (e.g. GradientBoostingClassifier)
        :param k: positive int, default 5, number of cross-validation folds
        :param metric: str, sci-kit learn metric to score the runs
        :return: array of cross-validation scores for each model run
        """
        # Time the cross-validation
        start = time.time()
        cv_score = cross_val_score(model, self.scalar.transform(train_features), train_target, cv=k, scoring=metric,
                                   verbose=2)
        print(f'Time to solve: {time.time() - start} seconds')
        print(f'Mean score: {cv_score.mean()}')
        print(f'STD of scores: {cv_score.std()}')
        return cv_score

    def train_model(self, train_features, train_target, model=None):
        """
        Trains a model, whether inputted or already defined in pipeline attribute

        :param train_features: pandas DataFrame object or array, matrix of training features
        :param train_target: pandas DataFrame object or array, vector of training target (or matrix if classification)
        :param model: object, def None, sci-kit learn model object to cross-validate (e.g. GradientBoostingClassifier)
        :return: Null, prints model training time
        """

        # Time model training and fit
        start = time.time()

        # If model is inputted as a parameter, train with that model
        if model is not None:
            self.pipeline = Pipeline([('scalar', self.scalar), ('ensemble', model)])
        self.pipeline.fit(train_features, train_target)
        print(f'Model trained in {(time.time() - start):.1f}')

    def test_model(self, test_features, test_label):
        """
        Tests the performance of model in pipeline attribute - specifically a REGRESSION model

        :param test_features: pandas DataFrame object or array, matrix of test features
        :param test_label: pandas DataFrame object or array, vector of test target
        :return: dictionary of r2, mean squared error and root mean squared error

        """

        # Predict on test set and return dictionary of performance scores
        y_hat = self.pipeline.predict(test_features)
        return {'r2': r2_score(y_hat, test_label),
                'mean_squared': mean_squared_error(y_hat, test_label),
                'root_mean_squared': np.sqrt(mean_squared_error(y_hat, test_label))}

    def test_model_clf(self, test_features, test_label):
        """
        Tests the performance of model in pipeline attribute - specifically a CLASSIFICATION model

        :param test_features: pandas DataFrame object or array, matrix of test features
        :param test_label: pandas DataFrame object or array, matrix of test target
        :return: dictionary of accuracy and ROC AUC score

        """

        # Predict on test set and return dictionary of performance scores
        y_hat = self.pipeline.predict(test_features)
        return {'accuracy': accuracy_score(y_hat, test_label),
                'roc': roc_auc_score(y_hat, test_label),
                # 'confusion_matrix': confusion_matrix(y_hat, test_label)
                }

    def save_model(self, file_name):
        """
        Save the model pipeline as a pickle file

        :param file_name: str, name of file with extension .pkl
        :return: Null
        """

        # Dump model into pickle file
        with open(file_name, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def save_scaler(self, file_name):
        """
        Save the model scaler as a pickle file

        :param file_name: str, name of file with extension .pkl
        :return: Null
        """

        # Dump scaler into pickle file
        with open(file_name, 'wb') as f:
            pickle.dump(self.scalar, f)

    def load_model(self, file_name):
        """
        Loads a model into the pipeline attribute

        :param file_name: str, name of file with extension .pkl:
        :return: Null

        """
        self.pipeline = pickle.load(open(file_name, 'rb'))


class FeedForwardNN(Module):
    def __init__(self, n_nodes, regression=True):
        """
        FeedForward Neural Network

        :param n_nodes: 3dim array of ints, units of each layer
        :param regression: bool, default=True, define whether the model is regression or classification

        """

        # Child class of Torch Module
        super(FeedForwardNN, self).__init__()
        self.REGRESSION = regression

        # Hidden Layer 1
        self.hidden_1 = Linear(n_nodes[0], n_nodes[1])
        self.function_1 = LeakyReLU()
        self.batch_1 = BatchNorm1d(n_nodes[1])

        # Hidden Layer 2
        self.hidden_2 = Linear(n_nodes[1], n_nodes[2])
        self.function_2 = LeakyReLU()
        self.batch_2 = BatchNorm1d(n_nodes[2])

        # Output Layer
        self.hidden_3 = Linear(n_nodes[2], (1 if regression else 3))

        # Initialization of weights
        xavier_uniform_(self.hidden_1.weight)
        xavier_uniform_(self.hidden_2.weight)
        xavier_uniform_(self.hidden_3.weight)

        self.scaler = MinMaxScaler()

    def fit_scaler(self, train_features):
        """
        Fits scalar attribute to train_features parameter
        :param train_features: pandas DataFrame object or array, matrix of training features
        :return: Null

        """

        return self.scaler.fit_transform(train_features)

    def forward(self, X):
        """
        Forward pass through Neural Network

        :param X: array matching dimensions of input
        :return: array, predicted value
        """

        X = self.function_1(self.hidden_1(X))
        X = self.batch_1(X)

        X = self.function_2(self.hidden_2(X))
        X = self.batch_2(X)
        return self.hidden_3(X)

    def train_model(self, train_data, test_data, learning_rate=0.001, num_epochs=100, beta1=0.9, beta2=0.999,
                    metric='mean_squared_error', mess=True):
        """
        :param train_data: torch data, training set
        :param test_data: torch data, validation set
        :param learning_rate: positive float, default 0.001, learning rate for Adams optimizer
        :param num_epochs: positive int, default 100, number of epochs to train model
        :param beta1: positive float, default 0.9, beta1 for Adams optimizer
        :param beta2: positive float, default 0.999, beta2 for Adams optimizer
        :param metric: str, default mean_squared_error, metric to evaluate performance
        :param mess: bool, default True
        :return:
        """

        # Define loss criterion and optimizer
        criterion = MSELoss() if self.REGRESSION else CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2))
        train_a, test_a = list(), list()

        for epoch in range(num_epochs):
            real, y_hats = list(), list()

            for (inputs, targets) in train_data:

                # Zero gradients and do forward pass
                optimizer.zero_grad()
                y_hat = self.forward(inputs)

                # Squeeze targets if classification
                targets = targets.squeeze() if self.REGRESSION is False else targets

                # Calculate loss and run Back-propagation
                loss = criterion(y_hat, (targets if self.REGRESSION else targets.long()))
                loss.backward()
                optimizer.step()

                # Calculate error
                y = targets.numpy()
                y_hats.append(y_hat.detach().numpy())
                real.append(y.reshape(len(y), 1))

            # Convert cross entropy value to categories
            y_hats = min_cross_entropy(y_hats) if self.REGRESSION is False else y_hats

            # Score training and validation data
            train_a.append(self.test_model(train_data)[metric])
            test_a.append(self.test_model(test_data)[metric])
            # train_a.append(mean_squared_error(np.vstack(real), np.vstack(y_hats))) \
            #     if self.REGRESSION else train_a.append(accuracy_score(np.vstack(real), np.vstack(y_hats)))

            # Print train and validation scores every 5 epochs
            if epoch % 5 == 0 and mess:
                print('Solved epoch number ' + str(epoch) + '!')
                print('Training Score: ' + str(train_a[epoch]))
                print('Validation Score: ' + str(test_a[epoch]))
        return np.array(train_a), np.array(test_a)

    def test_model(self, test_data):
        """
        :param test_data: torch data, data set to evaluate
        :return: dict, scores
        """

        y_hats, real = list(), list()

        # Loop through data and make forward pass to predict values
        for (inputs, targets) in test_data:
            y_hats.append(self.forward(inputs).detach().numpy())
            real.append(targets.numpy())

        # Convert cross entropy value to categories
        y_hats = min_cross_entropy(y_hats) if self.REGRESSION is False else y_hats

        return {'mean_squared_error': mean_squared_error(np.vstack(real), np.vstack(y_hats)),
                'r2': r2_score(np.vstack(real), np.vstack(y_hats))} \
            if self.REGRESSION else {'accuracy': accuracy_score(np.vstack(real), np.vstack(y_hats))}

    def predict(self, row):
        """
        :param row: array, input data to predict
        :return: array, prediction
        """
        return self.forward(Tensor(row)).detach().numpy()

    def save_model(self, path='my_model.pt'):
        """
        Save the model pipeline as a serialized file

        :param path: str, name of file with extension .pt
        :return: Null
        """

        save(self.state_dict(), path)

    def save_scaler(self, file_name):
        """
        Save the scaler as a pickle file

        :param file_name: str, name of file with extension .pkl
        :return: Null
        """

        with open(file_name, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self, path):
        """
        Loads a model into the pipeline attribute

        :param path: str, name of file with extension .pkl:
        :return: Null

        """

        self.load_state_dict(load(path))
        self.eval()

    def load_scaler(self, file_name):
        """
        :param file_name: str, name of file with extension .pkl
        :return: None, saves trained model as .pkl
        """
        self.scaler = pickle.load(open(file_name, 'rb'))

    # def forecast_price(self, predict_data):
    #     return np.array([self.forward(inputs).detach().numpy() for inputs in predict_data])
    #
    # def grid_search(self, domain, train_data, validate_data, lr=0.001, num_epochs=100, beta1=0.9, beta2=0.999):
    #     grid = [(i, j) for i in domain for j in domain]
    #     results_dic = dict()
    #
    #     for hidden_1, hidden_2 in grid:
    #         print('Solving ' + str(hidden_1) + 'x' + str(hidden_2))
    #         score_train, score_test = self.train_model(train_data, validate_data, learning_rate=lr,
    #                                                    num_epochs=num_epochs, beta1=beta1, beta2=beta2)
    #         results_dic.update({str(hidden_1) + 'x' + str(hidden_2): {'test': score_test[-1:],
    #                                                                   'train': score_train[-1:]}})
    #     return results_dic





