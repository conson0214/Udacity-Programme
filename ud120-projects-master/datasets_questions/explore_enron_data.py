#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
data = enron_data.values()
item = enron_data.items()
enron_data_poi = filter(lambda x:x[1]['poi']==True, item)

fastow_pay = enron_data['FASTOW ANDREW S']['total_payments']
skilling_pay = enron_data['SKILLING JEFFREY K']['total_payments']
lay_pay = enron_data['LAY KENNETH L']['total_payments']

enron_data_salary = filter(lambda x:x[1]['salary']!='NaN', item)
enron_data_known_address = filter(lambda x:x[1]['email_address']!='NaN', item)
enron_data_no_pay = filter(lambda x:x[1]['total_payments']=='NaN', item)

enron_data_no_pay_poi = filter(lambda x:x[1]['total_payments']=='NaN' and x[1]['poi']==True, item)

print enron_data_poi


