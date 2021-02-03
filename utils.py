import numpy as np
import pandas as pd
import datetime


"""

Helper functions for models

"""


"""Functions for models.py"""


def load_predict_data(df):
    return df.astype.astype('float32')


def one_hot_encode(vector):
    b = np.zeros((vector.size, vector.max() + 1))
    b[np.arange(vector.size), vector] = 1
    return b


def min_cross_entropy(vector):
    classes = list()
    for bat in vector:
        for row in bat:
            classes.append(np.argmax(row))
    return classes


"""Functions for features.py"""


def load_data(df, col):
    return pd.DataFrame(data=df[col].values.ravel(), columns=[col],
                        index=pd.date_range(datetime.date(2017, 10, 1), datetime.date(2021, 1, 20), freq='H'))


def load_query_data(df, col, sub_dict):
    df = df.loc[df[sub_dict['col']] == sub_dict['rows']][col]
    return pd.DataFrame(data=df.values.ravel(), columns=[col],
                        index=pd.date_range(datetime.date(2017, 10, 1), datetime.date(2021, 1, 20), freq='H'))


def clf_labels():
    df = pd.read_csv('lib/data/output_dispatch.csv', index_col=0)[['LI_chr', 'LI_dis']].copy()
    for c in ['chr', 'dis']:
        df.loc[df['LI_' + c] > 0, 'LI_' + c] = 1
        df['LI_' + c] = df['LI_' + c].astype(int)
    df['LI_none'] = np.where((df['LI_chr'] <= 0) & (df['LI_dis'] <= 0), 1, 0).astype('int')
    df.index = pd.date_range(datetime.date(2017, 10, 1), datetime.date(2021, 1, 20), freq='H')
    return df


def load_fuel_data():
    f = pd.read_csv('lib/data/fuel_data-FRSCE1.csv', index_col=0)
    f.index = pd.to_datetime(f.index)
    return f


"""Functions for training models"""


def parse_fuel_price(previous_day=True):
    df = pd.read_csv('lib/data/fuel_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index) + datetime.timedelta(days=(1 if previous_day else 0))
    return df.loc[df['Fuel Region'] == 'SCE1'].groupby('OPR_DT').mean()['Price'].append(
        df.loc[df['Fuel Region'] == 'FRSCE1'].groupby('OPR_DT').mean()['Price'])


def nn_clf_labels():
    df = pd.read_csv('lib/data/output_dispatch.csv', index_col=0)[['LI_chr', 'LI_dis']].copy()
    df['LI_op'] = 0
    for c, cls in zip(['chr', 'dis'], [1, 2]):
        df.loc[df['LI_' + c] > 0, 'LI_op'] = cls
    return pd.DataFrame(data=np.array(df['LI_op'].astype(int)), columns=['LI_op'],
                        index=pd.date_range(datetime.date(2017, 10, 1), datetime.date(2021, 1, 20), freq='H'))



