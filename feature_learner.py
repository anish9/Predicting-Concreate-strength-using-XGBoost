import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import trainer
from trainer import x_test,y_test
from trainer import data_

model_load = pickle.load(open("xgr",'rb'))

current_model = model_load



test_ = current_model.predict(x_test)

"""evaluating the R2Score"""


acc = r2_score(y_test,test_)

print("R2 SCORE : {}".format(acc))

mapp = {}
inv = {}
fi = current_model.feature_importances_
col = data_.columns[:-1]
for i in range(len(fi)):
    mapp[col[i]] = fi[i]
for i in range(len(fi)):
    inv[fi[i]] = col[i]

opd = sorted(mapp.values(),reverse=True)
print("FEATURES THAT MAKES IMPACT ON ACCURACY:")
print("----------------------------------------")
for i in opd:
    print(inv[i])