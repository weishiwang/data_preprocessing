# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:30:07 2022

@author: PC
"""


import pandas as pd
import numpy as np
from numpy import *
#from loadData import *

logp=pd.read_csv(r'logp数据.csv')
#对摩根指纹进行处理的函数
def chuli(x,y):
    return x[y]
# 只留下0、1
logp.摩根指纹 = logp.摩根指纹.astype('str')
logp.摩根指纹 = logp.摩根指纹.apply(lambda x:x.replace('.',''))
logp.摩根指纹 = logp.摩根指纹.apply(lambda x:x.replace('[',''))
logp.摩根指纹 = logp.摩根指纹.apply(lambda x:x.replace(']',''))
logp.摩根指纹 = logp.摩根指纹.apply(lambda x:x.replace(' ',''))

#单独生成一列数据
for i in range(len(logp.摩根指纹[0])):
    logp[i]=logp.apply(lambda x: chuli(x['摩根指纹'],i),axis=1)


for i in range(len(logp.摩根指纹[0])):
    logp[i] = logp[i].str[:].astype('float')
#print(logp.dtypes)
data=logp
data=data.drop('ID',axis=1)
data=data.drop('摩根指纹',axis=1)
data=data.drop('LogP值',axis=1)


def pca(dataMat, topNfeat=9999999):
    # 1 计算每一列的均值
    meanVals = mean(data, axis=0) # axis=0表示列，axis=1表示行
    print('各列的均值：\n', meanVals)

    # 2 去平均值，每个向量同时都减去均值
    meanRemoved = data - meanVals
    print('每个向量同时都减去均值:\n', meanRemoved)

    # 3 计算协方差矩阵的特征值与特征向量，eigVals为特征值， eigVects为
    # rowvar=0，传入的数据一行代表一个样本，若非0，传入的数据一列代表一个样本
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    print('特征值:\n', eigVals,'\n特征向量:\n', eigVects)

    # 4 将特征值排序, 特征值的逆序就可以得到topNfeat个最大的特征向量
    eigValInd = argsort(eigVals) # 特征值从小到大的排序，返回从小到大的index序号
#    print('eigValInd1=', eigValInd)

    # 5 保留前N个特征。-1表示倒序，返回topN的特征值[-1到-(topNfeat+1)不包括-(topNfeat+1)]
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # 重组 eigVects 最大到最小
    redEigVects = eigVects[:, eigValInd]
    print('重组n特征向量最大到最小:\n', redEigVects.T)

    # 6 将数据转换到新空间
    # print( "---", shape(meanRemoved), shape(redEigVects))
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # print('lowDDataMat=', lowDDataMat)
    # print('reconMat=', reconMat)
    return lowDDataMat, reconMat




if __name__ == "__main__":

    # 2 主成分分析降维特征向量设置
    lowDmat, reconMat = pca(data,1)
    print('PCA降维前的数据规模如下:\n',shape(data))
    print('PCA降维后的数据规模如下:\n',shape(lowDmat))

    