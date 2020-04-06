import os

import PIL
import cv2
import face_recognition
import numpy as np
import pandas as pd


# "'%06d'"%4
from PIL import Image


def extract_face(image):
    '''
    This function extract coordinates of face in image and then crop image

    :param image: image as ndarray
    :return: PIL Image of extracted face
             or False, if no face found
    '''
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        im = Image.fromarray(face_image)
        return im
    else:
        return False


def extract_faces_and_labels_for_video(path_to_video, path_to_extracted_frames, path_to_timesteps, new_image_size, path_to_existing_timesteps=''):
    '''
    This function takes video from path_to_video and extract faces from each frame of video with timesteps

    :param path_to_video: path to video file
    :param path_to_extracted_frames: path to folder, where extracted images should store
    :param path_to_timesteps: path to folder, where timesteps of extracted images should store
    :param new_image_size: size for extracted image
    :return:
    '''
    cap = cv2.VideoCapture(path_to_video)
    frame_rate = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS)
    total_frames = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
    timesteps_dataframe=pd.DataFrame(columns=['timestep','frame'], data=np.zeros(shape=(total_frames,2)))
    timesteps_dataframe['frame']=timesteps_dataframe['frame'].astype('str')
    iter=0
    step=1./frame_rate
    timestep=0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face=extract_face(frame)
        if face==False:
            timesteps_dataframe.iloc[iter, 0] = timestep
            timesteps_dataframe.iloc[iter, 1] = 'NO_FACE'
        else:
            face=face.resize(size=new_image_size, resample=PIL.Image.BILINEAR)
            filename=path_to_video.split('\\')[-1].split('.')[0]+'_'+"%05d"%iter
            face.save(path_to_extracted_frames +filename+'.png')
            timesteps_dataframe.iloc[iter,0]=timestep
            timesteps_dataframe.iloc[iter, 1]=filename
        timestep+=step
        iter+=1
    if path_to_existing_timesteps!='':
        tmp_timesteps=pd.read_csv(path_to_existing_timesteps+path_to_video.split('\\')[-1].split('.')[0]+'.csv', sep=';')
        timesteps_dataframe['timestep']=tmp_timesteps['time in seconds']
    timesteps_dataframe.to_csv(path_to_timesteps+path_to_video.split('\\')[-1].split('.')[0]+'.csv', index=False)

def preprocess_all_database(path_to_videos, path_to_extracted_frames, path_to_timesteps, new_image_size, path_to_existing_timesteps):
    '''
    :param path_to_videos: path to folder with videos
    :param path_to_extracted_frames: path to folder, where extracted images should store
    :param path_to_timesteps: path to folder, where timesteps of extracted images should store
    :param new_image_size: size for extracted image
    :param path_to_existing_timesteps: path to existing timesteps of database, if it exists
    :return:
    '''
    files = os.listdir(path_to_videos)
    if not os.path.exists(path_to_timesteps): os.mkdir(path_to_timesteps)
    if not os.path.exists(path_to_extracted_frames): os.mkdir(path_to_extracted_frames)
    for file in files:
        if not os.path.exists(path_to_extracted_frames+file.split('.')[0]+'\\'): os.mkdir(path_to_extracted_frames+file.split('.')[0]+'\\')
        extract_faces_and_labels_for_video(path_to_video=path_to_videos+file,
                                           path_to_extracted_frames=path_to_extracted_frames+file.split('.')[0]+'\\',
                                           path_to_timesteps=path_to_timesteps,
                                           new_image_size=new_image_size,
                                           path_to_existing_timesteps=path_to_existing_timesteps)


def concatenate_labels(path_to_full_labels_arousal,path_to_full_labels_valence, path_to_timesteps, path_to_save_complete_labels):
    # TODO: dodelay
    full_labels_arousal=pd.read_csv(path_to_full_labels_arousal)
    full_labels_arousal.columns=['filename_timestep','arousal']
    full_labels_arousal['filename'], full_labels_arousal['timestep'] = full_labels_arousal['filename_timestep'].str.split('_').str
    full_labels_arousal.drop(columns=['filename_timestep'], inplace=True)

    full_labels_valence=pd.read_csv(path_to_full_labels_valence)
    path_to_full_labels_valence.columns=['filename_timestep','valence']
    full_labels = pd.concat((full_labels_arousal, full_labels_valence['valence']))

    files = os.listdir(path_to_timesteps)
    for file in files:
        timesteps_dataframe=pd.read_csv(path_to_timesteps+file)
        filename='REC'+"%04d"%int(timesteps_dataframe['frame'].iloc[0].split('_')[0][1:]) # TODO: проверить, как досчитается
        tmp_df=pd.DataFrame(full_labels[full_labels['filename']==filename])
        # TODO: теперь по индексам (или другим способом) взять только те строки, таймстепы которых есть и там, и там
        



path_to_videos='D:\\DB\\RECOLA\\original\\RECOLA_Video_recordings\\1\\'
path_to_extracted_frames='D:\\DB\\RECOLA\\processed\\data\\'
path_to_timesteps='D:\\DB\\RECOLA\\processed\\timesteps\\'
path_to_existing_timesteps='D:\\DB\\RECOLA\\original\\RECOLA-Video-timings\\'
preprocess_all_database(path_to_videos, path_to_extracted_frames, path_to_timesteps, (224,224), path_to_existing_timesteps)