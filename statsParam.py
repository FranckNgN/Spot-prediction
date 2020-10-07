import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
import pandas as pd

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

def corrParam(df):
    """[summary]

    Args:
        df ([type]): [Unstacked DF]

    Returns:
        [df.pandas, mask.numpy]: [Parameter for the seaborn heatmap]
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr,dtype=np.bool))
    return corr, mask

def corrPlot(corr, mask):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    return sns.heatmap(corr, mask=mask,cmap=cmap,vmax = 1, center=0, vmin =-1, square=True, linewidths=.5, cbar_kws={"shrink": .5})