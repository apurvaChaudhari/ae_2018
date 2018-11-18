import pandas as pd

pd.set_option
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from data_processor import data_cleaner_train7 as data_cleaner_train
from data_processor import data_cleaner_test7 as data_cleaner_test
from data_processor import report

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)

PREFIX = 'gb3_'

pca_file_name = 'input_data/outputs/' + PREFIX + 'pca.save'
scaler_filename = 'input_data/' + PREFIX + 'scaler.save'
model_filename = 'input_data/' + PREFIX + 'model.save'
op_path = 'input_data/outputs/' + PREFIX + 'test_results.csv'
scale = StandardScaler()


# Model with rank: 1
# Mean validation score: 0.570 (std: 0.007)
# Parameters: {'n_estimators': 100, 'min_samples_split': 500, 'min_samples_leaf': 500, 'max_depth': 10}
#
# Model with rank: 2
# Mean validation score: 0.564 (std: 0.008)
# Parameters: {'n_estimators': 100, 'min_samples_split': 100, 'min_samples_leaf': 300, 'max_depth': 25}
#
# Model with rank: 3
# Mean validation score: 0.563 (std: 0.007)
# Parameters: {'n_estimators': 400, 'min_samples_split': 300, 'min_samples_leaf': 900, 'max_depth': 20}

def train_run():
    l_reg = GradientBoostingClassifier(max_features=None, subsample=0.80, random_state=0, verbose=True, learning_rate=0.05
                                       , n_estimators=300, min_samples_split=250, min_samples_leaf=250, max_depth=20)
    file_path = os.path.join(os.getcwd(), 'input_data/train_amex/train.csv')
    data = pd.read_csv(file_path)
    data, y_hat, pca = data_cleaner_train(data)
    data.loc[:, 'yhat'] = y_hat
    d1 = data.loc[data.yhat == 1, :]
    d2 = data.loc[data.yhat == 0, :]
    data_op = pd.DataFrame()

    for i in range(6):
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
    # param_dist = {'max_depth': range(10, 30,5), 'min_samples_split': range(100, 1000, 200), 'n_estimators': range(100, 1000, 300), 'min_samples_leaf': range(100, 1000, 200)}
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



def test_run():
    test_file_path = os.path.join(os.getcwd(), 'input_data/test_LNMuIYp/test.csv')
    data = pd.read_csv(test_file_path)
    output = pd.DataFrame()
    output.loc[:, 'session_id'] = data['session_id']

    model = joblib.load(model_filename)
    pca = joblib.load(pca_file_name)
    data = data_cleaner_test(data, pca)
    y_pred = model.predict(data.values)
    output.loc[:, 'is_click'] = y_pred
    output.to_csv(op_path, index=False)


if __name__ == '__main__':
    train_run()
    test_run()
