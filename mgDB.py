from pymongo import MongoClient
import pandas as pd
import os
from datetime import datetime

#Insert only unstacked data
#TODO: #check for both ? 

class mongoDB:
    def __init__(self):
        self.mongoLog = pd.read_csv(r'C:\Coding\Python Project\Stocks\Data\cmc.csv')['mongodb']

    def dbInsert(self, dbName, cltName, index, data):
        conn = MongoClient(self.mongoLog)  #Start connection with database
        check = 0
        try:
            db = conn.get_database(dbName)
            collection = db[cltName]
            print("DB connection succesfull")
        except:
            print("DB connection failed")
            check = 1

        try:
            collection.insert_one({"index": index, "data": data.to_dict("records"), "name": os.getlogin()})
        except:
            print("Insertion failed, no data inserted. Please try again")
            check = 1
        print("Insertion failed, please check." if check else "Insertion successful, please check.")
        conn.close() #Close connection with database

    def dbGet(self, dbName, cltName, index):
        ###TODO: Needs to check if the data exists in the db or not
        conn = MongoClient(self.mongoLog) #Start connection with database
        db = conn.get_database(dbName)
        collection = db[cltName]

        mongoDf = collection.find_one({"index": index})

        conn.close() #Close connection with database

        return pd.DataFrame(mongoDf["data"])#.drop("index", axis=1)#to check if index is date or not


# dbName = 'Finance'
# cltName = 'Price'
# index = 'Spot'

# db = mongoDB()
# df = pd.read_csv(r'C:\Coding\Python Project\Stocks\Data\xd.csv')
# db.dbInsert(dbName=dbName, cltName=cltName, index='Spot2066now', data=dtmgt.unstackData(df))
# db.dbGet(dbName=dbName, cltName=cltName, index=index)
# x =db.dbGet(dbName=dbName, cltName=cltName, index=index)