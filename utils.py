
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


# Data preprocessing 


# import the dataset file
FILE_PATH = os.path.join(os.getcwd(), 'housing.csv')
df_housing = pd.read_csv(FILE_PATH)

# replace the (<1H OCEAN) with (1H OCEAN)
df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN', inplace=True)

# Making new features
df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedrooms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
df_housing['pioulation_per_household'] = df_housing['population'] / df_housing['households']


# X = features, and y = target
X = df_housing.drop(columns='median_house_value', axis=1)
y = df_housing['median_house_value']

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.15, random_state=42)



num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int32', 'int64', 'float32', 'float64']]
categ_cols = [col for col in X_train.columns if X_train[col].dtype not in ['int32', 'int64', 'float32', 'float64']]


num_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(num_cols)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categ_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(categ_cols)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('OHE', OneHotEncoder(sparse=False))
])

total_pipeline = FeatureUnion(transformer_list=[
    ('num_pipe', num_pipeline),
    ('categ_pipe', categ_pipeline)
])

X_train_final = total_pipeline.fit_transform(X_train)


def preprocess_new(X_new):
    '''
    This function will process new instance before predicte using model
    
    Atgs:
        X_new: The new features to make data preprocessing on them
    
    Returns:
        Preprocessed features ready to enter the model
    '''
    return total_pipeline.transform(X_new)


