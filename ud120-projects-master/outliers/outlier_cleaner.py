#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    error = abs(predictions - net_worths)
    cleaned_data_raw = zip(ages, net_worths, error)
    cleaned_data_raw = sorted(cleaned_data_raw, key=lambda clean_data: clean_data[2])   # sorted by error

    cleaned_data = cleaned_data_raw[:int(len(cleaned_data_raw)*0.9)]

    return cleaned_data

