import os
import time

import PIL
import cv2
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image


def load_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))

def extract_face(image):
    '''
    This function extract coordinates of face in image and then crop image

    :param image: image as ndarray
    :return: PIL Image of extracted face
             or False, if no face found
    '''
    face_locations = face_recognition.face_locations(image)
    if len(face_locations)==1:
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        im = Image.fromarray(face_image)
        return im
    else:
        return len(face_locations)



def extract_cropped_faces_from_video(filename_video, path_to_boxes, path_to_labels, path_to_save_faces='', path_to_save_labels='', new_size=(224,224)):
    # TODO: Во всех случаях нет некоторых файлов .pts, нужно обрабатывать такие случаи (вычеркивать их из лейблов)
    # TODO: Думаю, также нужно записывать таймстепы...чтобы потом было возможно смешать данные с другими базами
    # TODO: ПРоверь, сколько всего кадров в видео и сколько лэйблов

    # videofile params
    cap = cv2.VideoCapture(filename_video)
    frame_rate = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS)
    total_frames = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
    # extract params
    timestep=1./frame_rate
    idx_frame = 0
    current_time=0
    format_video=filename_video.split('.')[-1]
    # arrays
    labels=pd.DataFrame(columns=['timestep','frame','arousal', 'valence'], data=np.zeros((total_frames,4)))
    labels['frame']=labels['frame'].astype('str')
    arousal=np.loadtxt(path_to_labels+'arousal\\'+filename_video.split('\\')[-1].split('.')[0]+'.txt').reshape((-1,))
    valence=np.loadtxt(path_to_labels+'valence\\'+filename_video.split('\\')[-1].split('.')[0]+'.txt').reshape((-1,))
    # extracting frames and annotations
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if format_video=='mp4' or format_video=='avi':
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if os.path.exists(path_to_boxes+str(idx_frame)+'.pts'):
            pts=load_pts(path_to_boxes+str(idx_frame)+'.pts')
            x,x_plus,y,y_plus=int(pts[0,0]), int(pts[2,0]), int(pts[0,1]), int(pts[2,1])
            img=Image.fromarray(frame)
            img=img.crop((x,y,x_plus,y_plus))
            img=img.resize(new_size, resample=PIL.Image.BILINEAR)
            frame_filename=filename_video.split('\\')[-1].split('.')[0]+'_'+"%05d"%idx_frame+'.png'
            img.save(path_to_save_faces+frame_filename)
            labels.iloc[idx_frame, 1] = frame_filename
        else:
            labels.iloc[idx_frame, 1] = 'NO_FACE'
        # labels
        labels.iloc[idx_frame, 0]=current_time
        labels.iloc[idx_frame, 2]=arousal[idx_frame]
        labels.iloc[idx_frame, 3]=valence[idx_frame]
        idx_frame += 1
    labels.to_csv(path_to_save_labels+filename_video.split('\\')[-1].split('.')[0]+'.csv', index=False)


def extract_cropped_faces_from_video_with_face_extractor(filename_video, path_to_labels, path_to_boxes, path_to_save_faces='', path_to_save_labels='', new_size=(224,224)):
    # videofile params
    cap = cv2.VideoCapture(filename_video)
    frame_rate = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS)
    total_frames = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
    # extract params
    timestep=1./frame_rate
    idx_frame = 0
    current_time=0
    format_video=filename_video.split('.')[-1]
    # arrays
    labels=pd.DataFrame(columns=['timestep','frame','arousal', 'valence'], data=np.zeros((total_frames,4)))
    labels['frame']=labels['frame'].astype('str')
    arousal=np.loadtxt(path_to_labels+'arousal\\'+filename_video.split('\\')[-1].split('.')[0]+'.txt').reshape((-1,))
    valence=np.loadtxt(path_to_labels+'valence\\'+filename_video.split('\\')[-1].split('.')[0]+'.txt').reshape((-1,))
    # extracting frames and annotations
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if format_video=='mp4' or format_video=='avi':
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=extract_face(frame)
        if img!=1:
            if os.path.exists(path_to_boxes + str(idx_frame) + '.pts'):
                pts = load_pts(path_to_boxes + str(idx_frame) + '.pts')
                x, x_plus, y, y_plus = int(pts[0, 0]), int(pts[2, 0]), int(pts[0, 1]), int(pts[2, 1])
                img = Image.fromarray(frame)
                img = img.crop((x, y, x_plus, y_plus))
                img = img.resize(new_size, resample=PIL.Image.BILINEAR)
                frame_filename = filename_video.split('\\')[-1].split('.')[0] + '_' + "%05d" % idx_frame + '.png'
                img.save(path_to_save_faces + frame_filename)
                labels.iloc[idx_frame, 1] = frame_filename
            else:
                labels.iloc[idx_frame, 1] = 'NO_FACE'
        else:
            img = img.resize(new_size, resample=PIL.Image.BILINEAR)
            frame_filename = filename_video.split('\\')[-1].split('.')[0] + '_' + "%05d" % idx_frame + '.png'
            img.save(path_to_save_faces + frame_filename)
            labels.iloc[idx_frame, 1] = frame_filename
        # labels
        labels.iloc[idx_frame, 0]=current_time
        labels.iloc[idx_frame, 2]=arousal[idx_frame]
        labels.iloc[idx_frame, 3]=valence[idx_frame]
        idx_frame += 1
        current_time+=timestep
    labels.to_csv(path_to_save_labels+filename_video.split('\\')[-1].split('.')[0]+'.csv', index=False)

def process_database(path_to_data, path_to_boxes, path_to_labels, path_to_save):
    videofiles=os.listdir(path_to_data)
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if not os.path.exists(path_to_save+'final_labels'):
        os.mkdir(path_to_save+'final_labels')
    if not os.path.exists(path_to_save + 'data'):
        os.mkdir(path_to_save + 'data')
    num=0
    for videofile in videofiles:
        start=time.time()
        if not os.path.exists(path_to_save+'data\\'+videofile.split('.')[0]):
            os.mkdir(path_to_save+'data\\'+videofile.split('.')[0])
        extract_cropped_faces_from_video_with_face_extractor(filename_video=path_to_data+videofile,
                                                             path_to_boxes=path_to_boxes+videofile.split('.')[0]+'\\',
                                                             path_to_labels=path_to_labels,
                                                             path_to_save_faces=path_to_save+'data\\'+videofile.split('.')[0]+'\\',
                                                             path_to_save_labels=path_to_save+'final_labels\\',
                                                             new_size=(224,224))
        end=time.time()
        print('videofile', videofile, 'processed...', 'num:',num, 'total videofiles:', len(videofiles), 'processed time:',end-start)
        num+=1


if __name__ == "__main__":
    '''    filename_video='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\Aff_wild\\Videos\\train\\309.mp4'
    path_to_boxes='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\Aff_wild\\bboxes\\train\\309\\'
    path_to_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\Aff_wild\\annotations\\train\\'
    path_to_save_frames='D:\\Databases\\Aff_wild\\processed\\309\\'
    path_to_save_labels = 'D:\\Databases\\Aff_wild\\processed\\'
    new_size=(224,224)
    extract_cropped_faces_from_video_with_face_extractor(filename_video=filename_video,
                                     #path_to_boxes=path_to_boxes,
                                     path_to_labels=path_to_labels,
                                     path_to_save_faces=path_to_save_frames,
                                     path_to_save_labels=path_to_save_labels,
                                     new_size=(224,224))'''
    process_database(path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\Aff_wild\\Videos\\train\\',
                     path_to_boxes='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\Aff_wild\\bboxes\\train\\',
                     path_to_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\Aff_wild\\annotations\\train\\',
                     path_to_save='D:\\Databases\\Aff_wild\\processed\\')