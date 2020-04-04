import os

import pandas as pd

path_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\zip\\validation.csv'
path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\Manually_Annotated_Images\\'
path_to_new_label='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\validation\\'
labels=pd.read_csv(path_labels)

labels=labels[labels.iloc[:,-1]!=-2]

counter=0
for i in range(labels.shape[0]):
    if os.path.exists(path_to_data+labels.iloc[i,0]):
        counter+=1

print(counter)
print(labels.shape[0])