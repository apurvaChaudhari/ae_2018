import pandas as pd

pd.set_option
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from data_processor import data_cleaner_train2, data_cleaner_test2, report
import pickle
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)

PREFIX = 'ten_'
is_int_fname = 'input_data/outputs/' + PREFIX + 'is_interesting.pickle'
to_keep_file_name = 'input_data/outputs/' + PREFIX + 'to_keep_prod_2.pickle'
scaler_filename = 'input_data/' + PREFIX + 'scaler.save'
model_filename = 'input_data/' + PREFIX + 'model.save'
op_path = 'input_data/outputs/' + PREFIX + 'test_results.csv'
scale = StandardScaler()

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(61, activation=None),
        keras.layers.Dense(200, activation=tf.nn.relu),
        keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_run():
    file_path = os.path.join(os.getcwd(), 'input_data/train_amex/train.csv')
    data = pd.read_csv(file_path)

    data, y_hat, scaler, to_keep_prod_2, = data_cleaner_train2(data)
    with open(to_keep_file_name, 'wb') as f:
        pickle.dump(to_keep_prod_2, f)

    data.loc[:, 'yhat'] = y_hat
    d1 = data.loc[data.yhat == 1, :]
    d2 = data.loc[data.yhat == 0, :]

    data_op = pd.DataFrame()

    for i in range(5):
        print(i)
        d1_sample = d1.sample(n=10000, replace=False)
        d2_sample = d2.sample(n=len(d1_sample), replace=False)
        data_tmp = d1_sample.append(d2_sample)
        data_op = data_op.append(data_tmp, ignore_index=True)
    data_op = data_op.sample(frac=1)
    yhat_op = data_op.loc[:, 'yhat']
    data_op.drop('yhat', axis=1, inplace=True)


    # predictors = list(data_op.drop('yhat', axis=1).columns)
    # modelfit(l_reg, data_op, predictors, performCV=True, printFeatureImportance=True, cv_folds=5)
    model=build_model()
    model.fit(data_op.values, yhat_op.values, epochs=1)

    data.drop('yhat', axis=1, inplace=True)
    output = model.predict_classes(data.values)
    roc = roc_auc_score(y_hat.values, output.ravel())
    print(roc)
    # joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    return model



def test_run(model):
    test_file_path = os.path.join(os.getcwd(), 'input_data/test_LNMuIYp/test.csv')
    data = pd.read_csv(test_file_path)
    output = pd.DataFrame()
    output.loc[:, 'session_id'] = data['session_id']
    scaler = joblib.load(scaler_filename)
    # model = joblib.load(model_filename)

    to_keep = pickle.load(open(to_keep_file_name, "rb"))
    data = data_cleaner_test2(data, scaler, to_keep)
    y_pred = model.predict(data.values)
    output.loc[:, 'is_click'] = y_pred
    output.to_csv(op_path, index=False)
    a = 10


if __name__ == '__main__':
    model=train_run()
    test_run(model)
