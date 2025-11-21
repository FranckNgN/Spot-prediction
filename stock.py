# import pandas as pd
# import numpy as np
# from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as  plt
import math
import pandas_datareader as web
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
beginDate = "2007-01-01"
# Read API key from environment for safety; set QUANDL_KEY in your environment
quandl_key = os.environ.get('QUANDL_KEY')
forecastDays = 30

BM = ['Copper','Nickel','Aluminium','Zinc']
PM = ['Gold','Silver','Palladium','Platinum']
commodities = BM + PM
underlying0 = ['GC=F','SI=F','PL=F','PA=F','HG=F','ALI=F','CL=F']#,'Silver','Platinum','Palladium','Copper','Nickel','Alluminium','Zinc','Crude Oil']
underlying = ['Gold','Silver','Platinum','Palladium','Copper','Aluminium','Crude Oil']

df = web.DataReader(underlying0,data_source = 'yahoo',start =beginDate,end="2020-04-09",api_key = quandl_key)
df=df['Close']
df = df.rename(columns ={'GC=F':'Gold','SI=F': 'Silver','PL=F':'Platinum','PA=F':'Palladium','HG=F':'Copper','ALI=F':'Aluminium','CL=F':'Crude Oil'}).dropna()

X = np.array(df)[:-forecastDays]
df[['Prediction '+ i for i in underlying]] = df[[i for i in underlying]].shift(-forecastDays)
y = np.array(df[['Prediction '+ i for i in underlying]])
y = y[:-forecastDays]
#Split data between 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Create and train the support vector machine (regressor)
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_rbf.fit(x_train, y_train)

#Testing mode: Score return the coefficient of determination R^2 of the prediction
#The best possible score is 1.0
#svm_confidence = svr_rbf.score(x_test, y_test)
#print("svm confidence: ",svm_confidence)

#Create and train the linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)

lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ",lr_confidence)

#Set x_forecast equal to the last 30 rows of the original data set from close price
x_forecast = np.array(df.drop(['Prediction '+  i for i in underlying],1))[-forecastDays:]

#print the linear model prediction for the next forecast days
lr_prediction = lr.predict(x_forecast)

#print the support vector prediction for the next forecast days
#svr_rbf_prediction = svr_rbf.predict(x_forecast)

df.loc[-forecastDays:,['Prediction ' + i for i in underlying]] = lr_prediction


df.to_csv(r'D:\Python Project\Commodities\Data\commodities1.csv')
