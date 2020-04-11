import os

import pandas as pd
import numpy as np

path_to_data_SEMAINE=''
path_to_labels_SEMAINE=''
path_to_data_RECOLA=''
path_to_labels_RECOLA=''
path_to_data_SEWA=''
path_to_labels_SEWA=''

def how_many_windows_do_you_need_for_this_labels(labels, size_of_window, step):
    # TODO: check it
    length=labels.shape[0]
    start=0
    how_many_do_you_need=0
    while True:
        if start+step>length:
            break
        start+=step
        how_many_do_you_need+=1
    if start!=length-1: how_many_do_you_need+=1
    return how_many_do_you_need


def load_labels(path_to_labels, size_window, step):
    # TODO: you need to end this function
    files=os.listdir(path_to_labels)
    for file in files:
        labels = pd.read_csv(path_to_labels + file)
        if 'SEW' in file or 'SEM' in file:
            labels=labels.iloc[::2] # to make timestep equal 0.04 (originally is 0.02) # TODO: check it
            num_windows=how_many_windows_do_you_need_for_this_labels(labels, size_window, step)
    pass

def transform_labels_to_windowed_labels(labels, size_window, step):
    # TODO: implement this function
    pass