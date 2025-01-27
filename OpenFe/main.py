# import sys
# sys.path.append('../')
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from openfe import OpenFE, tree_to_formula, transform
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


# Load files
train = pd.read_csv('/Users/aarononosala/Downloads/geoai-ground-level-no2-estimation-challenge20240612-4943-16iro0r/Train.csv')
test = pd.read_csv('/Users/aarononosala/Downloads/geoai-ground-level-no2-estimation-challenge20240612-4943-16iro0r/Test.csv')
test1 = test.copy()


data = train.dropna()

data['Date'] = pd.to_datetime(data['Date']) 
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

data.drop(['ID_Zindi', 'ID','Date'], axis=1, inplace = True)


def get_score(train_x, test_x, train_y, test_y):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    mse = mean_squared_error(test_y, pred)
    score = np.sqrt(mse)
    return score


if __name__ == '__main__':
    n_jobs = 4
  
    label = data[['GT_NO2']]
    del data['GT_NO2']

    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
    # get baseline score
    score = get_score(train_x, test_x, train_y, test_y)
    print("The RMSE before feature generation is", score)
    # feature generation
    ofe = OpenFE()
    ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)

    # OpenFE recommends a list of new features. We include the top 10
    # generated features to see how they influence the model performance
    train_x, test_x = transform(train_x, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
    score = get_score(train_x, test_x, train_y, test_y)
    print("The RMSE after feature generation is", score)
    print("The top 10 generated features are")
    for feature in ofe.new_features_list[:10]:
        print(tree_to_formula(feature))
  
    print("****************************************************************************************************************************************************************************************")
    print(train_x.columns)
   
   # Test set
    # Preprocess the test dataset
    '''test['Date'] = pd.to_datetime(test['Date'])
    test['year'] = test['Date'].dt.year
    test['month'] = test['Date'].dt.month
    test['day'] = test['Date'].dt.day

    # Drop unnecessary columns from the test set
    test.drop(['ID_Zindi', 'ID', 'Date'], axis=1, inplace=True)
    # Apply the same feature transformations to the unseen data
    test_transformed = transform(test, None, ofe.new_features_list[:10], n_jobs=4)[0]'''
