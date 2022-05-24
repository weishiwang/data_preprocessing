# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:00:37 2022

@author: Administrator
"""

import pandas as pd
import numpy as np
import math
import random

data = pd.read_excel('data2Clean.xlsx')
col = data.columns.to_list() 
null = data.isnull()
sumnull = data.isnull().sum()