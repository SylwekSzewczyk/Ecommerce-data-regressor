# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:55:08 2019

@author: Sylwek Szewczyk
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class Regression:
    
    def __init__ (self, df):
        self.df = df 
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.regressor = None
        self.y_preds = None
    
    def showData(self):
        return self.df.describe()
    
    def analyzeData(self):
        sns.set_palette("Reds")
        sns.set_style('whitegrid')
        return sns.pairplot(self.df)
    
    def splitData(self, testsize):
        self.X = self.df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
        self.y = self.df['Yearly Amount Spent']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = testsize, random_state = 100)
    
    def solve (self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
    
    def predict(self):
        self.y_preds = self.regressor.predict(self.X_test)
        plt.scatter(self.y_test, self.y_preds)
        plt.xlabel('y test')
        plt.ylabel('predictions')
        
    def measure(self):
        print('MAE:', metrics.mean_absolute_error(self.y_test, self.y_preds))
        print('MSE:', metrics.mean_squared_error(self.y_test, self.y_preds))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(self.y_test, self.y_preds)))
    
    def interprete(self):
        coef = pd.DataFrame(self.regressor.coef_, self.X.columns)
        coef.columns = ['Coefficients']
        return coef
    
    @classmethod
    def getData(cls, data):
        return cls(df = pd.read_csv(data))

r = Regression.getData('Ecommerce Customers')
r.analyzeData()