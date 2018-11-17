import pandas as pd

pd.set_option
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

scale = StandardScaler()


def data_cleaner_train(data, user_logs):
    to_keep_prod_2 = list(data['product_category_2'].value_counts().iloc[:10].index)
    data.drop(['session_id', 'DateTime', 'user_id', 'campaign_id'], axis=1, inplace=True)
    data.loc[~data['product_category_2'].isin(to_keep_prod_2), 'product_category_2'] = 0
    data.loc[:, 'city_development_index'] = data['city_development_index'].fillna(data['city_development_index'].value_counts().index[0])
    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    user_logs.drop(['DateTime', 'user_id'], axis=1, inplace=True)
    user_logs = user_logs.loc[user_logs.action == 'interest']['product'].value_counts()
    user_logs.iloc[:] = list(reversed(range(len(user_logs))))
    user_logs = user_logs.reset_index()
    user_logs.columns = ['product', 'is_interesting']
    data = pd.merge(data, user_logs, on='product', how='outer')
    y_hat = data['is_click']
    data.drop('is_click', axis=1, inplace=True)
    data = pd.get_dummies(data, columns=data.columns)
    ss_scalar = scale.fit(data[data.columns].as_matrix())
    data[data.columns] = ss_scalar.transform(data[data.columns].values)
    return data, y_hat, ss_scalar,to_keep_prod_2, user_logs


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
