import os

import pandas as pd
from PIL import Image

def images_preprocessing(labels, path_to_data, path_to_new_data, path_to_new_label, new_size):
    '''
    This function clean labels from -2 valence and arousal values (-2 means that the image is not face)
    Then cropping and resizing remaining images and save as a new data and new labels for this data
    to destination path (path_to_new_data, path_to_new_label)
    :param labels: original file with labels
    :param path_to_data: path to data
    :param path_to_new_data: path for saving new cropped and resized data
    :param path_to_new_label: path for saving new labels for this new cropped and resized data
    :param new_size: new size of images for resizing
    '''
    if not os.path.exists(path_to_new_data):
        os.mkdir(path_to_new_data)
    new_labels=pd.DataFrame(columns=labels.columns, data=labels[labels['valence']!=-2].copy(deep=True))
    new_labels.set_index('subDirectory_filePath', inplace=True)
    new_labels.drop(columns=new_labels.columns.difference(['face_x','face_y','face_width','face_height', 'valence', 'arousal']), inplace=True)
    for i in range(new_labels.index.shape[0]):
        img=Image.open(path_to_data+new_labels.index[i])
        x,y,w,h=new_labels.iloc[i,0:4]
        img=img.crop((x,y,x+w,y+h))
        folder_for_image=new_labels.index[i].split('/')[0]+'/'
        if not os.path.exists(path_to_new_data+folder_for_image):
            os.mkdir(path_to_new_data+folder_for_image)
        new_filename=path_to_new_data+new_labels.index[i].split('.')[0]+'.png'
        img=img.resize(new_size)
        img.save(new_filename)
        if i%1000==0:
            print(i,' images is processed...')
        new_labels.index[i]=new_labels.index[i].split('.')[0]+'.png'
    new_labels.to_csv(path_to_new_label+'new_labels.csv')


path_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\zip\\training.csv'
path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\Manually_Annotated_Images\\'
path_to_new_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\resized\\'
path_to_new_label='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\'
labels=pd.read_csv(path_labels)
images_preprocessing(labels, path_to_data,path_to_new_data, path_to_new_label, (224, 224))

