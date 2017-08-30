#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

item = data_dict.items()
foutlier  = filter(lambda x:x[1]['salary'] > 2.5e7 and not(x[1]['salary'] == 'NaN'), item)
foutlier2 = filter(lambda x:x[1]['salary'] > 1e6 and not(x[1]['salary'] == 'NaN') and x[1]['bonus'] > 5e6 \
                            and not(x[1]['bonus'] == 'NaN'), item)

data_dict.pop('TOTAL', 0 )

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


