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
    timesteps_dataframe.to_csv(path_to_timesteps+path_to_video.split('\\')[-1].split('.')[0]+'.csv', index=False)

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

def concatenate_labels(path_to_full_labels_arousal,path_to_full_labels_valence, path_to_timesteps, path_to_save_complete_labels):
    # TODO: Check it!
    '''
    This function split global label file (and concatenate arousal and valence) with all filenames on different subfiles according to timesteps and filenames
    :param path_to_full_labels_arousal: path to global 100 Hz label file with arousal
    :param path_to_full_labels_valence: path to global 100 Hz label file with valence
    :param path_to_timesteps: path to folder, where exist the timesteps for each video
    :param path_to_save_complete_labels: path to folder, where result of this function will saved
    :return:
    '''
    if not os.path.exists(path_to_save_complete_labels): os.mkdir(path_to_save_complete_labels)
    full_labels_arousal=pd.read_csv(path_to_full_labels_arousal)
    full_labels_arousal.columns=['filename_timestep','arousal']
    full_labels_arousal['filename'], full_labels_arousal['timestep'] = full_labels_arousal['filename_timestep'].str.split('_').str
    full_labels_arousal.drop(columns=['filename_timestep'], inplace=True)

    full_labels_valence=pd.read_csv(path_to_full_labels_valence)
    full_labels_valence.columns=['filename_timestep','valence']
    full_labels = pd.concat((full_labels_arousal, full_labels_valence['valence']), axis=1)
    full_labels=full_labels[['filename', 'timestep', 'arousal', 'valence']]
    full_labels['timestep']=full_labels['timestep'].astype('float64')
    files = os.listdir(path_to_timesteps)
    for file in files:
        timesteps_dataframe=pd.read_csv(path_to_timesteps+file)
        timesteps_dataframe['timestep']=timesteps_dataframe['timestep'].astype('float64')
        timesteps_dataframe=timesteps_dataframe.round({'timestep':2}) # rounding because of artifacts of float
        timesteps_dataframe.set_index('timestep', inplace=True)
        filename=file.split('.')[0]
        tmp_df=pd.DataFrame(full_labels[full_labels['filename']==filename])
        tmp_df.set_index('timestep', inplace=True)
        final=pd.merge(timesteps_dataframe, tmp_df, left_index=True, right_index=True)
        final.drop(columns=['filename'], inplace=True)
        final.to_csv(path_to_save_complete_labels+file)



path_to_videos='D:\\DB\\SEMAINE\\original\\SEMAINE_videos\\3\\'
path_to_extracted_frames='D:\\DB\\SEMAINE\\processed\\data\\'
path_to_timesteps='D:\\DB\\SEMAINE\\processed\\timesteps\\'
preprocess_all_database(path_to_videos, path_to_extracted_frames, path_to_timesteps, (224,224))

'''path_to_full_labels_arousal='D:\\DB\\SEMAINE\\SEM_labels_arousal_100Hz_gold_shifted.csv'
path_to_full_labels_valence='D:\\DB\\SEMAINE\\SEM_labels_valence_100Hz_gold_shifted.csv'
path_to_timesteps='D:\\DB\\SEMAINE\\processed\\timesteps\\'
path_to_save_complete_labels='D:\\DB\\SEMAINE\\processed\\final_labels\\'
concatenate_labels(path_to_full_labels_arousal, path_to_full_labels_valence, path_to_timesteps, path_to_save_complete_labels)'''