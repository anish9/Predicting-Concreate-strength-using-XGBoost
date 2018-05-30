""" a code on diagnosing feature importance using xgboost algorithm"""

""" developed for understanding the importance of features and its autoextraction using a machine learning """

#import the dependencies

import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

data = pd.read_csv("Concrete_Data.csv")

colu = {}
names = ["cement","blast_furnace","flyash","water","superplasticizer","coarses","fine","age","concreatecs"]
col = data.columns
for i in range(len(col)):
    colu[col[i]] = names[i]
data_ = data.rename(columns=colu)

x = data_.iloc[:,:8].values
y = data_.iloc[:,8].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
fit_xgr = XGBRegressor(max_depth=10,learning_rate=0.3,n_estimators=190)
fit_xgr = fit_xgr.fit(x_train,y_train)
pred_xgr=fit_xgr.predict(x_test)
#acc = r2_score(y_test,pred_xgr)



#saving the model for future prediction

pickle.dump(fit_xgr,open("xgr",'wb'))


if __name__ == "__main__":
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)	