import pandas as pd

pd.set_option
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from data_processor import data_cleaner_train6 as data_cleaner_train
from data_processor import data_cleaner_test6 as data_cleaner_test
from data_processor import report
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import scipy.stats as st

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)

PREFIX = 'xgb_'
pca_file_name = 'input_data/outputs/' + PREFIX + 'pca.save'
scaler_filename = 'input_data/' + PREFIX + 'scaler.save'
model_filename = 'input_data/' + PREFIX + 'model.save'
op_path = 'input_data/outputs/' + PREFIX + 'test_results.csv'
scale = StandardScaler()


# Model with rank: 1
# Mean validation score: 0.649 (std: 0.002)
# Parameters: {'subsample= 1.0,silent= False, 'reg_lambda= 1.0, 'n_estimators= 200, 'min_child_weight= 1.0, 'max_depth= 15,
# 'learning_rate= 0.1, 'gamma= 0.5, 'colsample_bytree= 0.9, 'colsample_bylevel= 0.9}
#
# Model with rank: 2
# Mean validation score: 0.643 (std: 0.003)
# Parameters: {'subsample= 0.8, 'silent= False, 'reg_lambda= 5.0, 'n_estimators= 800, 'min_child_weight= 1.0, 'max_depth= 15,
# 'learning_rate= 0.03, 'gamma= 0.5, 'colsample_bytree= 0.9, 'colsample_bylevel= 0.7}
#
# Model with rank: 3
# Mean validation score: 0.643 (std: 0.002)
# Parameters: {'subsample= 0.7, 'silent= False, 'reg_lambda= 1.0, 'n_estimators= 800, 'min_child_weight= 0.5, 'max_depth= 10,
# 'learning_rate= 0.03, 'gamma= 0.25, 'colsample_bytree= 0.8, 'colsample_bylevel= 0.9}

def train_run():
    l_reg = XGBClassifier(seed=27, silent=False, verbose_eval=True, subsample=0.8,  reg_lambda=5.0, n_estimators=800, min_child_weight=1, max_depth=15,
                          learning_rate=0.03, gamma=0.5, colsample_bytree=0.9, colsample_bylevel=0.7)

    file_path = os.path.join(os.getcwd(), 'input_data/train_amex/train.csv')
    data = pd.read_csv(file_path)
    data, y_hat, scaler, pca = data_cleaner_train(data)
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

    # # specify parameters and distributions to sample from
    # one_to_left = st.beta(10, 1)
    # from_zero_positive = st.expon(0, 50)
    #
    # param_dist = {
    #     'silent': [False],
    #     'max_depth': [15, 10,15],
    #     'learning_rate': [0.03, 0.05, 0.1],
    #     'subsample': [0.7, 0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.7, 0.8, 0.9],
    #     'colsample_bylevel': [0.7, 0.8, 0.9],
    #     'min_child_weight': [0.5, 1.0, 3.0, 7.0, 10.0],
    #     'gamma': [0.25, 0.5, 1.0],
    #     'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0],
    #     'n_estimators': [200, 500, 800]}
    #
    # # run randomized search
    # n_iter_search = 10
    # random_search = RandomizedSearchCV(l_reg, param_distributions=param_dist,
    #                                    n_iter=n_iter_search, cv=3, scoring='roc_auc', n_jobs=3)
    #
    # model = random_search.fit(data_op.values, yhat_op.values)
    # report(random_search.cv_results_)

    model = l_reg.fit(data_op.values, yhat_op.values)

    output = model.predict(data.values)

    roc = roc_auc_score(y_hat.values, output)
    print('#############################################################')
    print(roc)
    joblib.dump(pca, pca_file_name)
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)


def test_run():
    test_file_path = os.path.join(os.getcwd(), 'input_data/test_LNMuIYp/test.csv')
    data = pd.read_csv(test_file_path)
    output = pd.DataFrame()
    output.loc[:, 'session_id'] = data['session_id']
    scaler = joblib.load(scaler_filename)
    model = joblib.load(model_filename)
    pca = joblib.load(pca_file_name)
    data = data_cleaner_test(data, scaler, pca)
    y_pred = model.predict(data.values)
    output.loc[:, 'is_click'] = y_pred
    output.to_csv(op_path, index=False)


if __name__ == '__main__':
    train_run()
    test_run()
