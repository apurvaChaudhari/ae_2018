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
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

scale = StandardScaler()

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)

scaler_filename = "input_data/gb_scaler.save"
model_filename = "input_data/gb_model.save"
op_path='input_data/outputs/gb_test_results.csv'

def data_cleaner_train(data):
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id', 'product_category_2'], axis=1, inplace=True)
    data.loc[:, 'city_development_index'] = data['city_development_index'].fillna(data['city_development_index'].value_counts().index[0])
    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    ss_scalar = scale.fit(data[data.columns].as_matrix())
    data[data.columns] = ss_scalar.transform(data[data.columns].values)
    return data, y_hat, ss_scalar


def data_cleaner_test(data, scaler):
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id', 'product_category_2'], axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    imp_freq = SimpleImputer(strategy='most_frequent')
    data[data.columns] = imp_freq.fit_transform(data[data.columns].values)
    data[data.columns] = scaler.transform(data[data.columns].values)
    return data


def train_run():
    l_reg = GradientBoostingClassifier(learning_rate=1.0,n_estimators=300, max_depth=20, random_state=0)
    file_path = os.path.join(os.getcwd(), 'input_data/train_amex/train.csv')
    data = pd.read_csv(file_path)
    data, y_hat, scaler = data_cleaner_train(data)
    data.loc[:, 'yhat'] = y_hat
    d1 = data.loc[data.yhat == 1, :]
    d2 = data.loc[data.yhat == 0, :]

    data_op=pd.DataFrame()
    yhat_op = pd.DataFrame()
    # TODO make a load balancer !!!!!!!
    for i in range(30):
        print(i)
        d1_sample = d1.sample(n=10000, replace=True)
        d2_sample = d2.sample(n=len(d1_sample), replace=True)
        data_tmp = d1_sample.append(d2_sample)
        data_op=data_op.append(data_tmp,ignore_index=True)

    yhat_op = data_op.loc[:, 'yhat']
    data_op.drop('yhat', axis=1, inplace=True)
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
    data = data_cleaner_test(data, scaler)
    y_pred = model.predict(data.values)
    output.loc[:, 'is_click'] = y_pred
    output.to_csv(op_path, index=False)
    a = 10


if __name__ == '__main__':
    train_run()
    test_run()
