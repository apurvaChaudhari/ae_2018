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
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)

PREFIX = 'gb_sp_'
is_int_fname = 'input_data/outputs/' + PREFIX + 'is_interesting.pickle'
to_keep_file_name = 'input_data/outputs/' + PREFIX + 'to_keep_prod_2.pickle'
scaler_filename = 'input_data/' + PREFIX + 'scaler.save'
model_filename = 'input_data/' + PREFIX + 'model.save'
op_path = 'input_data/outputs/' + PREFIX + 'test_results.csv'
scale = StandardScaler()


def train_run():
    xgb = GradientBoostingClassifier(max_features=None, subsample=0.80, random_state=0, verbose=True,
                                     learning_rate=0.05, n_estimators=200, min_samples_split=500, min_samples_leaf=500, max_depth=15)


    file_path = os.path.join(os.getcwd(), 'input_data/other/train2.csv')
    data = pd.read_csv(file_path)

    yhat_op = data.loc[:, 'is_click']
    data.drop(['is_click', 'session_id'], axis=1, inplace=True)
    data = pd.get_dummies(data,columns=data.columns)

    model = xgb.fit(data.values, yhat_op.values)

    output = model.predict(data.values)

    roc = roc_auc_score(yhat_op.values, output)
    print('#############################################################')
    print(roc)
    joblib.dump(model, model_filename)


def test_run():
    test_file_path = os.path.join(os.getcwd(), 'input_data/other/test2.csv')
    data = pd.read_csv(test_file_path)
    output = pd.DataFrame()
    output.loc[:, 'session_id'] = data['session_id']
    model = joblib.load(model_filename)
    y_pred = model.predict(data.values)
    output.loc[:, 'is_click'] = y_pred
    output.to_csv(op_path, index=False)


if __name__ == '__main__':
    train_run()
    test_run()
