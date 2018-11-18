import pandas as pd

pd.set_option
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import cross_val_score
from datetime import *
import matplotlib.pylab as plt
from sklearn import decomposition
from matplotlib.pylab import rcParams
from sklearn import manifold

rcParams['figure.figsize'] = 12, 4
scale = StandardScaler()

filter_list = ['product_category_2_0.0',
               'product_category_2_143597.0',
               'product_category_2_146115.0',
               'product_category_2_168114.0',
               'product_category_2_18595.0',
               'product_category_2_234846.0',
               'product_category_2_235358.0',
               'product_category_2_254132.0',
               'product_category_2_255689.0',
               'product_category_2_269093.0',
               'product_category_2_270915.0',
               'product_category_2_32026.0',
               'product_category_2_327439.0',
               'product_category_2_408790.0',
               'product_category_2_419804.0',
               'product_category_2_447834.0',
               'product_category_2_450184.0',
               'product_category_2_82527.0']


def data_cleaner_train(data, user_logs):
    to_keep_prod_2 = list(data['product_category_2'].value_counts().iloc[:5].index)
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id'], axis=1, inplace=True)
    data.loc[~data['product_category_2'].isin(to_keep_prod_2), 'product_category_2'] = 0
    data.loc[:, 'city_development_index'] = data['city_development_index'].fillna(data['city_development_index'].value_counts().index[0])
    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    user_logs.drop(['DateTime', 'user_id'], axis=1, inplace=True)
    user_logs = user_logs.loc[user_logs.action == 'interest']['product'].value_counts().iloc[:3]
    user_logs.iloc[:] = list(reversed(range(len(user_logs))))
    user_logs = user_logs.reset_index()
    user_logs.columns = ['product', 'is_interesting']
    data = pd.merge(data, user_logs, on='product', how='outer')
    data.loc[:, 'is_interesting'] = data['is_interesting'].fillna(0)
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    ss_scalar = scale.fit(data[data.columns].as_matrix())
    data[data.columns] = ss_scalar.transform(data[data.columns].values)
    return data, y_hat, ss_scalar, to_keep_prod_2, user_logs


def data_cleaner_test(data, scaler, to_keep_prod_2, modified_user_logs):
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id'], axis=1, inplace=True)
    data.loc[~data['product_category_2'].isin(to_keep_prod_2), 'product_category_2'] = 0
    data.reset_index(drop=True, inplace=True)
    data = pd.merge(data, modified_user_logs, on='product', how='outer')
    data = pd.get_dummies(data, columns=data.columns)
    imp_freq = SimpleImputer(strategy='most_frequent')
    data[data.columns] = imp_freq.fit_transform(data[data.columns].values)
    data[data.columns] = scaler.transform(data[data.columns].values)
    return data


def data_cleaner_train2(data):
    to_keep_prod_2 = list(data['product_category_2'].value_counts().iloc[:5].index)
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id'], axis=1, inplace=True)
    data.loc[~data['product_category_2'].isin(to_keep_prod_2), 'product_category_2'] = 0
    data.loc[:, 'city_development_index'] = data['city_development_index'].fillna(data['city_development_index'].value_counts().index[0])
    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    # user_logs.drop(['DateTime', 'user_id'], axis=1, inplace=True)
    # user_logs = user_logs.loc[user_logs.action == 'interest']['product'].value_counts()
    # user_logs.iloc[:] = list(reversed(range(len(user_logs))))
    # user_logs = user_logs.reset_index()
    # user_logs.columns = ['product', 'is_interesting']
    # data = pd.merge(data, user_logs, on='product', how='outer')
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    ss_scalar = scale.fit(data[data.columns].as_matrix())
    data[data.columns] = ss_scalar.transform(data[data.columns].values)
    return data, y_hat, ss_scalar, to_keep_prod_2


def data_cleaner_test2(data, scaler, to_keep_prod_2):
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id'], axis=1, inplace=True)
    data.loc[~data['product_category_2'].isin(to_keep_prod_2), 'product_category_2'] = 0
    data.reset_index(drop=True, inplace=True)
    # data = pd.merge(data, modified_user_logs, on='product', how='outer')
    data = pd.get_dummies(data, columns=data.columns)
    imp_freq = SimpleImputer(strategy='most_frequent')
    data[data.columns] = imp_freq.fit_transform(data[data.columns].values)
    data[data.columns] = scaler.transform(data[data.columns].values)
    return data


def data_cleaner_train3(data):
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id', 'product_category_2'], axis=1, inplace=True)
    data = data.fillna('NAN')
    data.reset_index(drop=True, inplace=True)
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    ss_scalar = scale.fit(data[data.columns].as_matrix())
    data[data.columns] = ss_scalar.transform(data[data.columns].values)
    return data, y_hat, ss_scalar


def data_cleaner_test3(data, scaler):
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id', 'product_category_2'], axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = data.fillna('NAN')
    data = pd.get_dummies(data, columns=data.columns)
    data[data.columns] = scaler.transform(data[data.columns].values)
    return data


def data_cleaner_train4(data):
    data.drop(['session_id', 'user_id', 'product_category_2'], axis=1, inplace=True)

    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DateTime'] = [datetime.time(d) for d in data['DateTime']]
    conditions = [
        (data['DateTime'] < datetime(1900, 1, 1, 12).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 17).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 22).time())]
    choices = ['morning', 'afternoon', 'evening']
    data['DateTime'] = np.select(conditions, choices, default='late night')
    data = data.fillna('NAN')

    data.reset_index(drop=True, inplace=True)
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    ss_scalar = scale.fit(data[data.columns].as_matrix())
    data[data.columns] = ss_scalar.transform(data[data.columns].values)
    return data, y_hat, ss_scalar


def data_cleaner_test4(data, scaler):
    data.drop(['session_id', 'user_id', 'product_category_2'], axis=1, inplace=True)

    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DateTime'] = [datetime.time(d) for d in data['DateTime']]
    conditions = [
        (data['DateTime'] < datetime(1900, 1, 1, 12).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 17).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 22).time())]
    choices = ['morning', 'afternoon', 'evening']
    data['DateTime'] = np.select(conditions, choices, default='late night')
    data = data.fillna('NAN')

    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    data = data.fillna('NAN')
    data[data.columns] = scaler.transform(data[data.columns].values)
    return data


def data_cleaner_train5(data):
    data.drop(['session_id', 'user_id', 'campaign_id', 'product_category_2'], axis=1, inplace=True)

    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DateTime'] = [datetime.time(d) for d in data['DateTime']]
    conditions = [
        (data['DateTime'] < datetime(1900, 1, 1, 12).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 17).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 22).time())]
    choices = ['morning', 'afternoon', 'evening']
    data['DateTime'] = np.select(conditions, choices, default='late night')
    data = data.fillna('NAN')

    data.reset_index(drop=True, inplace=True)
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    ss_scalar = scale.fit(data[data.columns].as_matrix())
    data[data.columns] = ss_scalar.transform(data[data.columns].values)
    return data, y_hat, ss_scalar


def data_cleaner_test5(data, scaler):
    data.drop(['session_id', 'user_id', 'campaign_id', 'product_category_2'], axis=1, inplace=True)

    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DateTime'] = [datetime.time(d) for d in data['DateTime']]
    conditions = [
        (data['DateTime'] < datetime(1900, 1, 1, 12).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 17).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 22).time())]
    choices = ['morning', 'afternoon', 'evening']
    data['DateTime'] = np.select(conditions, choices, default='late night')
    data = data.fillna('NAN')

    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    data = data.fillna('NAN')
    data[data.columns] = scaler.transform(data[data.columns].values)
    return data


def data_cleaner_train6(data):
    data.drop(['session_id', 'user_id', 'product_category_2'], axis=1, inplace=True)

    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DateTime'] = [datetime.time(d) for d in data['DateTime']]
    conditions = [
        (data['DateTime'] < datetime(1900, 1, 1, 12).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 17).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 22).time())]
    choices = ['morning', 'afternoon', 'evening']
    data['DateTime'] = np.select(conditions, choices, default='late night')
    data = data.fillna(0)

    data.reset_index(drop=True, inplace=True)
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)

    ss_scalar = scale.fit(data[data.columns].as_matrix())
    data[data.columns] = ss_scalar.transform(data[data.columns].values)
    pca = decomposition.PCA(n_components=10)
    pca.fit(data)
    data = pca.transform(data)
    data = pd.DataFrame(data)

    return data, y_hat, ss_scalar, pca


def data_cleaner_test6(data, scaler, pca):
    data.drop(['session_id', 'user_id', 'product_category_2'], axis=1, inplace=True)

    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DateTime'] = [datetime.time(d) for d in data['DateTime']]
    conditions = [
        (data['DateTime'] < datetime(1900, 1, 1, 12).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 17).time()),
        (data['DateTime'] < datetime(1900, 1, 1, 22).time())]
    choices = ['morning', 'afternoon', 'evening']
    data['DateTime'] = np.select(conditions, choices, default='late night')
    data = data.fillna('NAN')

    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)

    data[data.columns] = scaler.transform(data[data.columns].values)
    data = pca.transform(data)
    data = pd.DataFrame(data)
    return data


def data_cleaner_train7(data):
    global filter_list
    data.drop(['session_id', 'user_id', 'DateTime'], axis=1, inplace=True)

    # data['DateTime'] = pd.to_datetime(data['DateTime'])
    # data['DateTime'] = [datetime.time(d) for d in data['DateTime']]
    # conditions = [
    #     (data['DateTime'] < datetime(1900, 1, 1, 12).time()),
    #     (data['DateTime'] < datetime(1900, 1, 1, 17).time()),
    #     (data['DateTime'] < datetime(1900, 1, 1, 22).time())]
    # choices = ['morning', 'afternoon', 'evening']
    # data['DateTime'] = np.select(conditions, choices, default='late night')
    data = data.fillna(0)

    data.reset_index(drop=True, inplace=True)
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data.loc[~data['product_category_2'].isin(filter_list), 'product_category_2'] = 0
    data = pd.get_dummies(data, columns=data.columns)

    # ss_scalar = scale.fit(data[data.columns].as_matrix())
    # data[data.columns] = ss_scalar.transform(data[data.columns].values)
    pca = decomposition.PCA(n_components=5)
    pca.fit(data)
    data = pca.transform(data)
    data = pd.DataFrame(data)
    # filter_list2 = [i for i in data.columns if 'product_category_2' in i]
    return data, y_hat, pca


def data_cleaner_test7(data, pca):
    global filter_list
    data.drop(['session_id', 'user_id', 'DateTime'], axis=1, inplace=True)

    # data['DateTime'] = pd.to_datetime(data['DateTime'])
    # data['DateTime'] = [datetime.time(d) for d in data['DateTime']]
    # conditions = [
    #     (data['DateTime'] < datetime(1900, 1, 1, 12).time()),
    #     (data['DateTime'] < datetime(1900, 1, 1, 17).time()),
    #     (data['DateTime'] < datetime(1900, 1, 1, 22).time())]
    # choices = ['morning', 'afternoon', 'evening']
    # data['DateTime'] = np.select(conditions, choices, default='late night')
    data = data.fillna(0)

    data.reset_index(drop=True, inplace=True)
    data.loc[~data['product_category_2'].isin(filter_list), 'product_category_2'] = 0
    data = pd.get_dummies(data, columns=data.columns)

    # data[data.columns] = scaler.transform(data[data.columns].values)
    data = pca.transform(data)
    data = pd.DataFrame(data)
    return data


def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['yhat'])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain['yhat'], cv=cv_folds, scoring='roc_auc')

    # Print model report:
    print("\nModel Report")
    print(
        "Accuracy : %.4g" % metrics.accuracy_score(dtrain['yhat'].values, dtrain_predictions))
    print(
        "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['yhat'], dtrain_predprob))

    if performCV:
        print
        "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
        return feat_imp


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
