import pandas as pd
import pandas_datareader as web
import os

#f(..,*args = [],**kwargs={})
"""
Franck B Nguyen 2020-02-19
REMARGE UNSTACKDATA AND UNSTACKDF
ISSUE WITH DATE
"""

# API keys should be provided via environment variables. Do NOT hardcode keys here.
key = None

def unstackData(df,typeList = [], underlying1List=[]):
    if typeList:
        df=df[df.type.isin(typeList)]
        
    if underlying1List:
        df=df[df.underlying1.isin(underlying1List)]
    df=df.pivot_table(index="date",columns="underlying1",values="price")
    df.index = pd.to_datetime(df.index, format = "%Y-%m-%d")
    return df
    
def stackDf(df,mapping):
    print('Reformating dataframe, might take a long time')
    mapping = mapping.dropna(subset=["YFcode"]).drop(["YFcode","YFname"],axis=1) #DROP NA AND INDEX COLUMN USED FOR API
    underlying0 = mapping.underlying.loc[0]
    x = df[['date', underlying0]].rename(columns={underlying0: 'price'}).dropna() #Drop na for missing value, gold
    info = mapping.loc[mapping.underlying == underlying0]
    for k in range(len(x) - 1):
        info = info.append(mapping.loc[mapping.underlying == underlying0])
    #Concatanation needs to be on a unique index
    x.reset_index(drop=True, inplace=True)
    info.reset_index(drop=True, inplace=True)
    data = pd.concat([info, x], axis=1)

    # Filling dataframe with all values except gold
    for i in mapping.underlying[1:]:
        x = df[['date', i]].rename(columns={i: 'price'}).dropna()#drop na for missing value for each underlying
        info = mapping.loc[mapping.underlying == i]
        for k in range(len(x) - 1):
            info = info.append(mapping.loc[mapping.underlying == i])
        x.reset_index(drop=True, inplace=True)
        info.reset_index(drop=True, inplace=True)
        data = data.append(pd.concat([info, x], axis=1))
    #data.index = data.date
    print("Stack format successful")

    return data.reset_index(drop=True)

class data:
    def __init__(self,strBeginDate=None,strEndDate = None,**kwargs):
        """
        :param strBeginDate: Begin date in string, format = "%Y-%m-%d"
        :param strEndDate: End Date in string, format = "%Y-%m-%d"
        :param kwargs: API key
        To do : add a split for api source and check the api key
                -Add something when symbol could not be read
        """
        #Set up api stock connection
        #Set range date
        self.beginDate = strBeginDate
        self.endDate =  strEndDate
        # Set API key from environment (or via kwargs). Do NOT hardcode keys in repository.
        # Priority: kwargs overrides environment variable
        self.quandlKey = kwargs.get('quandlKey') or os.environ.get('QUANDL_KEY')
        self.avKey = kwargs.get('avKey') or os.environ.get('AV_KEY')
        #Set underlying and split
        self.underlyingCSV = pd.read_csv(r'C:\Coding\Python Project\Stocks\Data\underlying.csv') #Get mapping csv file. to change to mongoDB
        self.CTY = self.underlyingCSV.loc[self.underlyingCSV.type == "CTY"]
        self.EQD = self.underlyingCSV.loc[self.underlyingCSV.type == "EQD"]
        self.FX = self.underlyingCSV.loc[self.underlyingCSV.type == "FX"]
        self.INDEX = self.underlyingCSV.loc[self.underlyingCSV.type == "INDEX"]
        self.underlying = self.underlyingCSV#Select underlying which as a yahoo finance code only within underlyingCSV

        #Get closed price data from yahoo finance for each underlying from YFcode if exists
    def underlyingFormat(self,underlying):
        """
        :param underlying: subset of underlyingCSV in pandas bulk dataframe
        :return: same dataframe without NA and only underlying and yahoo finance code
        """
        return underlying[['underlying','YFcode']].dropna(subset=["YFcode"])

    def getData(self,underlyingPandas):
        print("Get data")
        # if(self.quandlKey):
        #     df = web.DataReader(list(underlyingPandas.YFcode),data_source = 'yahoo',start =self.beginDate,api_key = self.quandlKey)['Close']
        # elif(self.avKey):
        #     df = web.DataReader(list(underlyingPandas.YFcode),data_source = 'yahoo',start =self.beginDate,api_key = self.avKey)['Close']
        # else:
        underlying = list(underlyingPandas.YFcode)
        noData =[]

        df = web.DataReader(underlying, data_source='yahoo', start=self.beginDate, end=self.endDate)['Close'] # get all in one list
        #df = web.DataReader(underlying[0], data_source='yahoo', start=self.beginDate, end=self.endDate)['Close']
        # for i in underlying[1:]:
        #     try:
        #         dfi = web.DataReader(i, data_source='yahoo', start=self.beginDate, end=self.endDate)['Close']
        #         df= df.append(dfi)
        #     except:
        #         noData.append(i)
        

        if noData : 
            print("Missing data for:",noData)
        print("Stock data retrieval sucessful")
        df.columns = [i for i in underlyingPandas.underlying]
        df['date'] = df.index

        return df
    