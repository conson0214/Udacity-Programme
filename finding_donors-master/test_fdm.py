# -*- coding: utf-8 -*-

# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time

# 导入人口普查数据
data = pd.read_csv("census.csv")

print data.head(n=1)

# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

income = income_raw
income.loc[income_raw == ">50K"] = 1
income.loc[income_raw == "<=50K"] = 0

print sum(income)

print(len(income))

