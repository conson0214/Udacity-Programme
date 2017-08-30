#-*- coding: utf-8 -*-

# 载入此项目所需要的库
import numpy as np
import pandas as pd
from sklearn import model_selection
import visuals as vs

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, prices, test_size=0.2, random_state=0)

# 完成
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

#TODO 1

#目标：计算价值的最小值
minimum_price = min(prices)

#目标：计算价值的最大值
maximum_price = max(prices)

#目标：计算价值的平均值
mean_price = np.mean(prices)

#目标：计算价值的中值
median_price = np.median(prices)

#目标：计算价值的标准差
std_price = np.std(prices)

#目标：输出计算的结果
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)




