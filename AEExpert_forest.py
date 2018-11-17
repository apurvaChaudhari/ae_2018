# Problem Statement
# Recent years have witnessed a surge in the number of internet savvy users. Companies in the financial
# services domain leverage this huge internet traffic arriving at their interface by strategically placing ads/promotions
# for cross selling of various financial products on a plethora of web pages. The digital analytics unit of Best Cards Company
#  uses cutting edge data science and machine learning for successful promotion of its valuable card products.
# They believe that a predictive model that forecasts whether a session involves a click on the ad/promotion would help them
# extract the maximum out of the huge clickstream data that they have collected. You are hired as a consultant to build an
# efficient model to predict whether a user will click on an ad or not, given the following features:

# Clickstream data/train data for duration: (2nd July 2017 – 7th July 2017)
# Test data for duration: (8th July 2017 – 9th July 2017)
# User features (demographics, user behaviour/activity, buying power etc.)
# Historical transactional data of the previous month with timestamp info (28th May 2017– 1st July 2017) (User views/interest registered)
# Ad features (product category, webpage, campaign for ad etc.)
# Date time features (exact timestamp of the user session)

# Variable-----------------------------------Definition
# session_id---------------------------------Unique ID for a session
# DateTime-----------------------------------Timestamp
# user_id------------------------------------Unique ID for user
# product------------------------------------Product ID
# campaign_id--------------------------------Unique ID for ad campaign
# webpage_id---------------------------------Webpage ID at which the ad is displayed
# product_category_1-------------------------Product category 1 (Ordered)
# product_category_2-------------------------Product category 2
# user_group_id------------------------------Customer segmentation ID
# gender-------------------------------------Gender of the user
# age_level----------------------------------Age level of the user
# user_depth---------------------------------Interaction level of user with the web platform (1 - low, 2 - medium, 3 - High)
# city_development_index---------------------Scaled development index of the residence city
# var_1--------------------------------------Anonymised session feature
# is_click-----------------------------------0 - no click, 1 - click


# HistoricalUser logs
#
# Variable-----------Definition
# DateTime-----	Timestamp
# user_id-----------Unique ID for the user
# product-----Product ID
# action----------------view/interest (view - viewed the product page, interest - registered interest for the product)


# Evaluation Metric
# The evaluation metric for this competition is AUC-ROC score.


import pandas as pd

pd.set_option
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from data_processor import data_cleaner_train, data_cleaner_test
import pickle

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)

PREFIX = 'forest_'
is_int_fname = 'input_data/outputs/' + PREFIX + 'is_interesting.pickle'
to_keep_file_name = 'input_data/outputs/' + PREFIX + 'to_keep_prod_2.pickle'
scaler_filename = 'input_data/' + PREFIX + 'scaler.save'
model_filename = 'input_data/' + PREFIX + 'model.save'
op_path = 'input_data/outputs/' + PREFIX + 'test_results.csv'
scale = StandardScaler()


def train_run():
    l_reg = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=0, n_jobs=7, verbose=2)
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
