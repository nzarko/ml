# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os

class ZReport :
    def __init__(self,df):
        self.df = df
        
    def nan_report(self,percent=None, view_total=False):
        nan_df = self.df.isna()
        nan_array = nan_df.iloc[:,:].values
        
        total_nans = np.sum(nan_array, axis=0)
        total = nan_array.shape[0]
        if view_total :
            total_array = np.full(shape=total_nans.shape[0],
                                  fill_value = total,
                                  dtype=np.int)
            
        total_array_percentage = np.around(total_nans/total * 100,4)
        
        if view_total :
            contents = {'Columns':self.df.columns,
                        'Nans Count':total_nans,
                        'Column Total':total_array,
                        'Nans Percentage':total_array_percentage}
        else:
            contents = {'Columns':self.df.columns,
                        'Nans Count':total_nans,                
                        'Nans Percentage':total_array_percentage}
        
        result = pd.DataFrame(contents).sort_values(['Nans Percentage'],
                           ascending=False)
        print(result)
        if percent is not None:
            return result[result['Nans Percentage'] >= 70]
        
        return result
    
    def dropna(self, min_percent=0.7):
        df = self.df.copy()
        out_df = df[[column for column in df if df[column].isna().mean() < min_percent]]
        print('List of dropped coluns : ', end=' ')
        for c in df.columns:
            if c not in out_df.columns:
                print(c, end=', ')
            
        return out_df
        
        
DATA_DIR='../input'
print(os.listdir(DATA_DIR))

train_df = pd.read_csv(DATA_DIR+'/train.csv')
test_df = pd.read_csv(DATA_DIR+'/test.csv')

# Test ZReport
zp = ZReport(train_df)
nr = zp.nan_report(percent=70)

out_df = train_df[[column for column in train_df if train_df[column].isna().mean() < 0.7]]
print('List of dropped coluns : ', end=' ')
for c in train_df.columns:
    if c not in out_df.columns:
        print(c, end=', ')
        
out_f = zp.dropna(min_percent=0.7)

