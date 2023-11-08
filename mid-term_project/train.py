
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split


df = pd.read_csv('/home/administrator/mlzoomcamp/project/WineQT.csv')


X_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(X_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.quality.values
y_val = df_val.quality.values
y_test = df_test.quality.values

del df_train['quality']
del df_val['quality']
del df_test['quality']



def train(df_train, y_train, C=10):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

# Save the model to a file
filename = 'model.bin'
dv = 'dv.bin'
with open(filename, 'wb') as file:
    pickle.dump(filename, file)

with open(dv, 'wb') as file:
    pickle.dump(dv, file)