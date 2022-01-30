# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from permetrics.regression import Metrics
from sklearn.preprocessing import LabelEncoder


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def encode_categorical(df, cols):
    
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        #not_null = df[col][df[col].notnull()]
        df[col] = df[col].fillna('nan')
        df[col] = pd.Series(le.fit_transform(df[col]), index=df.index)

    return df


def print_metrics(y_true, y_pred):
    obj1 = Metrics(y_true, y_pred)
    print("MAE {}".format(obj1.mean_absolute_error(clean=True, decimal=3)))
    print("MSE {}".format(obj1.mean_squared_error(clean=True, decimal=3)))
    print("RMSE {}".format(obj1.root_mean_squared_error(clean=True, decimal=3)))
    print("MAPE {}".format(obj1.mean_absolute_percentage_error(clean=True, decimal=3)))
    print("SMAPE {}".format(obj1.symmetric_mean_absolute_percentage_error(clean=True, decimal=3)))
    print("MAAPE {}".format(obj1.mean_arctangent_absolute_percentage_error(clean=True, decimal=3)))
    print("R2 {}".format(obj1.r2(clean=True, decimal=3)))
