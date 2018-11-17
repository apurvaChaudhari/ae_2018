import pandas as pd

pd.set_option
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from data_processor import data_cleaner_train, data_cleaner_test
import pickle

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)

PREFIX = 'gb_'
is_int_fname = 'input_data/outputs/' + PREFIX + 'is_interesting.pickle'
to_keep_file_name = 'input_data/outputs/' + PREFIX + 'to_keep_prod_2.pickle'
scaler_filename = 'input_data/' + PREFIX + 'scaler.save'
model_filename = 'input_data/' + PREFIX + 'model.save'
op_path = 'input_data/outputs/' + PREFIX + 'test_results.csv'
scale = StandardScaler()


def train_run():
    l_reg = GradientBoostingClassifier(learning_rate=1.0, n_estimators=300, max_depth=20, random_state=0, verbose=True)
    file_path = os.path.join(os.getcwd(), 'input_data/train_amex/train.csv')
    userlogs_path = os.path.join(os.getcwd(), 'input_data/train_amex/historical_user_logs.csv')
    data = pd.read_csv(file_path)
    userlogs = pd.read_csv(userlogs_path)

    data, y_hat, scaler, to_keep_prod_2, is_interesting = data_cleaner_train(data, userlogs)
    with open(to_keep_file_name, 'wb') as f:
        pickle.dump(to_keep_prod_2, f)
    is_interesting.to_pickle(is_int_fname)

    data.loc[:, 'yhat'] = y_hat
    d1 = data.loc[data.yhat == 1, :]
    d2 = data.loc[data.yhat == 0, :]

    data_op = pd.DataFrame()
    yhat_op = pd.DataFrame()

    for i in range(50):
        print(i)
        d1_sample = d1.sample(n=10000, replace=True)
        d2_sample = d2.sample(n=len(d1_sample), replace=True)
        data_tmp = d1_sample.append(d2_sample)
        data_op = data_op.append(data_tmp, ignore_index=True)
    yhat_op = data_op.loc[:, 'yhat']
    data_op.drop('yhat', axis=1, inplace=True)
    data.drop('yhat', axis=1, inplace=True)
    data = data.sample(frac=1)
    model = l_reg.fit(data_op.values, yhat_op.values)
    output = model.predict(data.values)
    # output[output > 0.5] = 1
    # output[output < 0.5] = 0
    roc = roc_auc_score(y_hat.values, output)
    print(roc)
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    a = 10


def test_run():
    test_file_path = os.path.join(os.getcwd(), 'input_data/test_LNMuIYp/test.csv')
    data = pd.read_csv(test_file_path)
    output = pd.DataFrame()
    output.loc[:, 'session_id'] = data['session_id']
    scaler = joblib.load(scaler_filename)
    model = joblib.load(model_filename)
    is_interesting = pd.read_pickle(is_int_fname)

    to_keep = pickle.load(open(to_keep_file_name, "rb"))
    data = data_cleaner_test(data, scaler, to_keep, is_interesting)
    y_pred = model.predict(data.values)
    output.loc[:, 'is_click'] = y_pred
    output.to_csv(op_path, index=False)
    a = 10


if __name__ == '__main__':
    train_run()
    test_run()
