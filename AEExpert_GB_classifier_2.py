import pandas as pd

pd.set_option
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from data_processor import data_cleaner_train5 as data_cleaner_train
from data_processor import data_cleaner_test5 as data_cleaner_test
from data_processor import report

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
    l_reg = GradientBoostingClassifier(max_features=None, subsample=0.80, random_state=0, verbose=True, learning_rate=0.05)
    file_path = os.path.join(os.getcwd(), 'input_data/train_amex/train.csv')
    data = pd.read_csv(file_path)
    data, y_hat, scaler = data_cleaner_train(data)
    data.loc[:, 'yhat'] = y_hat
    d1 = data.loc[data.yhat == 1, :]
    d2 = data.loc[data.yhat == 0, :]
    data_op = pd.DataFrame()

    for i in range(3):
        print(i)
        d1_sample = d1.sample(frac=1)
        d2_sample = d2.sample(n=len(d1_sample), replace=False)
        data_tmp = d1_sample.append(d2_sample)
        data_op = data_op.append(data_tmp, ignore_index=True)

    data_op = data_op.sample(frac=1)
    yhat_op = data_op.loc[:, 'yhat']

    data_op.drop('yhat', axis=1, inplace=True)
    data.drop('yhat', axis=1, inplace=True)

    # specify parameters and distributions to sample from
    param_dist = {'max_depth': range(10, 30), 'min_samples_split': range(100, 1000, 200), 'n_estimators': range(100, 1000, 300), 'min_samples_leaf': range(100, 1000, 200)}

    # run randomized search
    n_iter_search = 5
    random_search = RandomizedSearchCV(l_reg, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, scoring='roc_auc', n_jobs=2)

    model = random_search.fit(data_op.values, yhat_op.values)
    report(random_search.cv_results_)

    # model = l_reg.fit(data_op.values, yhat_op.values)

    output = model.predict(data.values)

    roc = roc_auc_score(y_hat.values, output)
    print('#############################################################')
    print(roc)
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)


def test_run():
    test_file_path = os.path.join(os.getcwd(), 'input_data/test_LNMuIYp/test.csv')
    data = pd.read_csv(test_file_path)
    output = pd.DataFrame()
    output.loc[:, 'session_id'] = data['session_id']
    scaler = joblib.load(scaler_filename)
    model = joblib.load(model_filename)
    data = data_cleaner_test(data, scaler)
    y_pred = model.predict(data.values)
    output.loc[:, 'is_click'] = y_pred
    output.to_csv(op_path, index=False)


if __name__ == '__main__':
    train_run()
    test_run()
