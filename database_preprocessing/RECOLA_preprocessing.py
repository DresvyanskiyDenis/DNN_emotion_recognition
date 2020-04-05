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


def extract_faces_and_labels_for_video(path_to_video, path_to_extracted_frames, path_to_timesteps, new_image_size):
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
    timesteps_dataframe.to_csv(path_to_timesteps+path_to_video.split('\\')[-1].split('.')[0]+'.csv')

def preprocess_all_database(path_to_videos, path_to_extracted_frames, path_to_timesteps, new_image_size):
    '''

    :param path_to_videos: path to folder with videos
    :param path_to_extracted_frames: path to folder, where extracted images should store
    :param path_to_timesteps: path to folder, where timesteps of extracted images should store
    :param new_image_size: size for extracted image
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
                                           new_image_size=new_image_size)


def concatenate_labels_(labels, my_own_labels):

    pass

path_to_video='D:\\DB\\001_RECOLA\\RECOLA_Video_recordings\\P16.mp4'
path_to_extracted_frames='D:\\DB\\001_RECOLA\\temp\\'
if not os.path.exists(path_to_extracted_frames):
    os.mkdir(path_to_extracted_frames)
path_to_timesteps='D:\\DB\\001_RECOLA\\temp\\'
extract_faces_and_labels_for_video(path_to_video, path_to_extracted_frames, path_to_timesteps, (224,224))