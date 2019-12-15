# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:55:08 2019

@author: Sylwek Szewczyk
"""

import pandas as pd
import seaborn as sns
import numpy as np

class Regression:
    
    def __init__ (self, df):
        self.df = df 
    
    def showData(self):
        return self.df.describe()
    
    def analyzeData(self):
        sns.set_palette("Reds")
        sns.set_style('whitegrid')
        return sns.pairplot(self.df)
    
    @classmethod
    def getData(cls, data):
        return cls(df = pd.read_csv(data))

r = Regression.getData('Ecommerce Customers')
r.analyzeData()