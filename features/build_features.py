import numpy as np
import pandas as pd
import datetime
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, "/Users/jhussman/Documents/predict-energy")
from util import load_data, load_query_data


class DataMatrix:
    def __init__(self, target, target_labels=None):
        """
        :param target: pandas DataFrame
        :param target_labels
        """
        self.df = target
        self.TARGET = [self.df.columns[0]] if target_labels is None else target_labels

    def add_fourier_terms(self, k=1):
        """
        :param k: int default 1, number of fourier terms
        :return: Null, updates DataFrame attribute
        """
        for k_i in range(1, k+1):
            # Sine terms
            self.df['day_sin_' + str(k_i)] = np.sin(2 * np.pi * k_i * self.df.index.hour / 24.)
            self.df['week_sin_' + str(k_i)] = np.sin(2 * np.pi * k_i * self.df.index.dayofweek / 7.)
            self.df['year_sin_' + str(k_i)] = np.sin(2 * np.pi * k_i * self.df.index.dayofyear / 365.25)

            # Cosine terms
            self.df['day_cos_' + str(k_i)] = np.cos(2 * np.pi * k_i * self.df.index.hour / 24.)
            self.df['week_cos_' + str(k_i)] = np.cos(2 * np.pi * k_i * self.df.index.dayofweek / 7.)
            self.df['year_cos_' + str(k_i)] = np.cos(2 * np.pi * k_i * self.df.index.dayofyear / 365.25)

    def add_previous_days(self, price, days=1):
        """
        :param price:
        :param days: int default 1, number of previous days to include as features
        :return: Null, updates DataFrame attribute
        """
        for day in range(1, days+1):
            # Add t minus 24 prices for specified number of days
            self.df['day_t_min_' + str(day)] = np.concatenate([np.full(24*day, np.nan), price[:-24*day].values.ravel()])

    def add_features(self, feature_dic):
        """
        :param feature_dic: dic, contains path and label info to load raw data from csv
        :return: Null, updates DataFrame attribute
        """
        for label, sub_dic in feature_dic.items():
            self.df[label] = load_query_data(pd.read_csv(sub_dic['path'], index_col=0), sub_dic['col'], sub_dic['q']) \
                if 'q' in sub_dic.keys() else load_data(pd.read_csv(sub_dic['path'], index_col=0), sub_dic['col'])

    def add_fuel_price(self, fuel_df):
        self.df['fuel_price'] = fuel_df.reindex(index=self.df.index, method='ffill')

    def add_clf(self):
        df = pd.read_csv('/Users/jhussman/Documents/predict-energy/lib/data/output_dispatch.csv')[['LI_soc', 'LI_previous']]
        df['LI_soc'] = df['LI_soc'] / df['LI_soc'].max()
        df.index = pd.date_range(datetime.date(2017, 10, 1), datetime.date(2021, 1, 20), freq='H')
        self.df['soc'] = df[['LI_soc']].copy()
        # self.df['previous'] = df[['LI_previous']].copy()

    def feature_names(self):
        """
        :return: list of str, list of feature names
        """
        return [f for f in self.df.columns if f not in self.TARGET]

    def corr_matrix(self):
        """
        :return: pandas DataFrame object, Correlation Matrix of Feature Importance based on PearsonR stat
        """
        copy_df = self.df.dropna().copy()
        corr_dic = dict()
        for col in self.feature_names():
            corr_dic.update({col: pearsonr(copy_df[col].dropna().values, copy_df[self.TARGET].values)[0]})

        return pd.DataFrame(data=corr_dic.values(), index=corr_dic.keys(), columns=['feature'])\
            .sort_values(by=['feature'], ascending=False)

    def plot_feature_importance(self):
        self.corr_matrix().abs().sort_values(by=['feature'], ascending=False).plot.bar()

    def baseline_performance(self):
        return {'r2': r2_score(self.df.dropna()[self.TARGET], self.df.dropna()['day_t_min_1']),
                'error': np.sqrt(mean_squared_error(self.df.dropna()[self.TARGET], self.df.dropna()['day_t_min_1']))}

    def split(self, train_size=0.9):
        """
        :param train_size: float between 0 and 1 defaulted at 0.9, percent of set as training data
        :return: split sets
        """
        return train_test_split(self.df.dropna(), shuffle=True, train_size=train_size)


class TorchDataSet(Dataset):
    def __init__(self, feature_matrix):
        self.MATRIX = feature_matrix
        self.n_inputs = len(self.MATRIX.df.columns) - 1
        self.n = len(self.MATRIX.df.dropna())
        self.scaler = MinMaxScaler()

        self.X = self.scaler.fit_transform(self.MATRIX.df.dropna()[self.MATRIX.feature_names()].values.astype('float32'))
        self.y = self.MATRIX.df.dropna()[self.MATRIX.TARGET[0]].values.astype('float32').reshape(self.n, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return [self.X[i], self.y[i]]

    def get_splits(self, validation_percent=0.05, test_percent=0.05):
        return random_split(self, [self.n - round(validation_percent * self.n) - round(test_percent * self.n),
                                   round(validation_percent * self.n), round(test_percent * self.n)])

    def process_splits(self, batch=64, validation_percent=0.05, test_percent=0.05):
        train, validate, test = self.get_splits(validation_percent, test_percent)
        return DataLoader(train, batch_size=batch, shuffle=True), DataLoader(validate, batch_size=batch, shuffle=False),\
            DataLoader(test, batch_size=batch, shuffle=False)



