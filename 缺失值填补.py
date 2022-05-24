# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:27:55 2022

@author: 01
"""
import pandas as pd
import numpy as np
import math
import random
#=======================方法一=================================================
data = pd.read_excel('data2Clean.xlsx')
col = data.columns.to_list()
null = data.isnull()
sumnull = data.isnull().sum()
#print(data.isnull().sum())
#Averagedec = data.Average.describe()
for i in range(len(sumnull)):

    sumnull = data.isnull().sum()
    if(sumnull[i]!=0):
        index = col[i]
        
        print('列名：',index)
        print('存在缺失值个数：',sumnull[i])
#        for i in range(sumnull[i]):
        tab = data[index].describe()
        tc= round(random.uniform(tab[4],tab[6]),5)#对应25%——75%之间产生随机数
#        data.fillna({index: tc})
        data.loc[:, index] = data[index].fillna(tc)
        sumnull[i]=0
#   
#index = sumnull.index[sumnull[]]
print()
print('缺失值填补完成')
print('查看是否还有缺失值')   
sumnull = data.isnull().sum()
print(data.isnull().sum())



#=======================方法二填补一列多个缺失值=================================================
#
#data = pd.read_excel('data2Clean.xlsx')
#col = data.columns.to_list()
#null = data.isnull()
#sumnull = data.isnull().sum()
#for i in range(len(sumnull)):
#
#    sumnull = data.isnull().sum()
#    index = col[i]
#    if(sumnull[i]>0):
#        print('在第',i+1,'列有缺失值：')
#        print('存在缺失值个数：',sumnull[i])
#        for j in range(sumnull[i]):
#            index1=data.index[np.where(np.isnan(data[index]))[0]]
#            print('列名：',index)
#            
#            print('第',index1,'行有缺失值：')
#            
#            tab = data[index].describe()
#            tc= round(random.uniform(tab[4],tab[6]),5)#对应25%——75%之间产生随机数
#            data[index][index1]=tc
#print()
#print('缺失值填补完成')
#print('查看是否还有缺失值')            
#sumnull = data.isnull().sum()
#print(data.isnull().sum())

