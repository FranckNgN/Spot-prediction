import dataMngt as dtmgt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import  seasonal_decompose
import pmdarima as pm
import numpy as np
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv(r'C:\Python Project\Commodities\Data\stocks.csv')

def RMSE(df1,df2):
    return np.sqrt(mean_squared_error(df1,df2))

def standardized(df):
    return (df-df.mean())/df.std()
def stationary_test(df):
    print('Results of Dickey-Fuller Test:')
    df_test=adfuller(df)
    indices = ['Test Statistic', 'p-value', 'No. Lags Used', 'Number of Observations Used']
    output = pd.Series(df_test[0:4], index=indices) # 4 lags for quarterly, 12 for monthly
    for key, value in df_test[4].items():
        output['Critical value (%s)' % key] = value
    print(output)

xau1 = dtmgt.unstackDf(df)#to change !!!!!
xau['Date']=xau.index
xau['Date'] = pd.to_datetime(xau['Date'])
xau.set_index('Date', inplace=True)
xau=xau[:-160]
train = xau[:-30]
test = xau[-30:]
ts_diff = train.diff(1).dropna() #d=1
result = xau.pct_change().dropna()*100
#stats.kstest(result, 'norm')


rolling_mean = xau.rolling(window = 30).mean()
rolling_std = xau.rolling(window = 12).std()
plt.plot(xau, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
#plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Gold price and its Rolling Mean')
plt.show()

#transform TS into stationnary TS
#Tendance : differenciation ordre 1

plt.plot(ts_diff)
plt.title('1st degree diff on gold')
stationary_test(ts_diff)

plot_acf(ts_diff)
plot_pacf(ts_diff)


plt.title("Histogramme résidu diff")
"""
H0 : Hypothese nulle, la serie temporelle n est pas stationnaire ( résidu dépend de lui même)
H1 : Hypothese 1, la serie temporelle est stationnaire, independante de elle même. Ce que l on désire

Test stats < CV10 < CV5 < CV1 < pValue
    On est sur a 99% que la serie est stationnaire
pValue = Proba H0 will not be rejected, si grand : on ne rejete pas donc stationnaire sinon on rejete
"""
from scipy import stats
#Check heteroscedasticite (Si les résidues sont stionnaire ou pas)
#Decomposition tendance/seasonalite / résidu
#pas la diff mais la serie elle meme et etudier les residues
decomposition = seasonal_decompose(ts_diff, period=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

result = seasonal_decompose(xau,period=30)
result.plot()
plt.show()
plot_pacf(residual.dropna())
plt.title("pacf residual")

plt.subplot(411)
plt.plot(ts_diff,label="Original")
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label="Tendance")
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label="Saisonalite")
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label="Residue")
plt.legend(loc='best')

#Check Résidue si stationnaire ou pas
stationary_test(residual.dropna())
"""
Test stats < CV1 < CV5 < CV10 < pValue => Residue stationnaire
Nous n avons aucune tendance dans les résidues
"""

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
            'corr':corr, 'minmax':minmax})

model = ARIMA(train, order=(1,1,1))
fit = model.fit(disp=0)
predict = fit.predict() #fit.predict(start='2019-07-30')
#plt.title("Actual vs fitted")
fc, se, conf = fit.forecast(30, alpha=0.05)
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)


fc_series = pd.Series(fc, index=test.index)

print("RMSE ARIMA(2,1,2) : ",np.sqrt(mean_squared_error(test,fc)))

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Gold forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


model = ARIMA(ts_diff, order=(0, 1, 0))#p,d,q
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')


model = pm.auto_arima(ts_diff.values, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=30,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)


fit1 = SARIMAX(xau, order=(2, 1, 1),seasonal_order=(0,1,1,7)).fit()
pred = fit1.predict(start = "2019-08-29",end="2019-10-10")
plt.figure(figsize=(16,8))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(pred, label='SARIMA')
plt.legend(loc='best')
plt.show()

print(np.sqrt(mean_squared_error(test,pred)))

stats.kstest(sresid,cdf="t")



#Prediction série temporelle
plot_acf(ts_diff) #p=15
plot_pacf(ts_diff) #q = 15
res = []
p=5
q=5
d=3
for i in range(p):
    for j in range(q):
        for k in range(d):
            try:
                print("p:",i,"q:",j,"d:",d)
                model = ARIMA(train, order=(i,j,k))
                fit = model.fit()
                fc = fit.forecast(30, alpha=0.05)[0]
                res.append([i,j,k,np.sqrt(mean_squared_error(test,fc))])
            except:
                print("Failed:","p:", i, "q:", j,"d:",k)
l =pd.DataFrame(res)
l.columns = ['AR','MA','Diff','RMSE']
print(l[l.RMSE==l.RMSE.min()])


#--------------------------GARCH--------------------------------
#On suppose série stationnaire
from arch import arch_model
import pandas as pd
arimaResid = model.fit().resid
type = ['GARCH','ARCH','EGARCH']#, 'FIARCH','HARCH'
law = ['generalized error','skewt','normal','studentst']
am = arch_model(arimaResid, vol='GARCH', p=1, q=1, dist='generalized error')  # boucle sur les type de garch + type de loi
fit = am.fit()
residuals = pd.DataFrame()  # ljung box sur les residus + kolmogorov

for i in type:
    for j in law:
        am = arch_model(ts_diff,vol=i,p=2,q=2,dist=j)#boucle sur les type de garch + type de loi
        fit = am.fit()
        resi = fit.resid#ljung box sur les residus + kolmogorov
        residuals[i+' '+j] = resim

#residuals = standardized(residuals)
#forecast = fit.forecast(horizon=30)

#Ljung Box : test d'absance d'autocorelation entre les résidus
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_result=None
ljung_result = pd.DataFrame()
ljung_result['type'] = 'GARCH generalized error'

for i in residuals.columns:
    ljungIndex = acorr_ljungbox(residuals[i],return_df=True,lags=10)
    ljungIndex['type'] = i
    ljung_result = ljung_result.append(ljungIndex)




#for i in range(len(ljung)):
    #print("lag :",i,"stat value:",round(ljung.iloc[i].lb_stat,2),"p-value:",round(ljung.iloc[i].lb_pvalue,2))

#am = arch_model(fit.resid,vol='GARCH',p=2,q=2,dist='StudentsT')
#test Engle : test heteroscedascticite
from statsmodels.stats.diagnostic import het_arch as engle
engle_result = None
engle_result = pd.DataFrame()

for i in residuals.columns:
    engle_index = pd.DataFrame(engle(residuals[i]))
    engle_index['type'] = i
    engle_result = engle_result.append(engle_index)



test = xau
test['pct_change'] = test.pct_change().dropna()
test['stdev30'] = test['pct_change'].rolling(30).std()
test['hvol30'] = test['stdev30']*(252**0.5) # Annualize.
test['variance'] = test['hvol30']**2
test = test.dropna() # Remove rows with blank cells.

returns = test['pct_change']
am = arch_model(returns)
res = am.fit()
test['C'] = res.params['omega']
test['B'] = test['variance'] * res.params['beta[1]']
test['A'] = (test['pct_change']**2) * res.params['alpha[1]']
test['forecast_var'] = test.loc[:,'C':'A'].sum(axis=1)
test['forecast_vol'] = test['forecast_var']**0.5

mu_pred = fit.forecast()[0]
et_pred = am.forecast(horizon=1).mean['h.1'].iloc[-1]



#------------LJUNG BOX TEST
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung = acorr_ljungbox(fit.resid**2,return_df=True,lags=7)
print("Ljung Box test sur les résidus carrés")
for i in range(len(ljung)):
    print("lag :",i,"stat value:",round(ljung.iloc[i].lb_stat,2),"p-value:",round(ljung.iloc[i].lb_pvalue,2))

residueA=fit.res
residueA.index = pd.to_datetime(residueA.index)
residueA.index.to_datetime()