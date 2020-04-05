import PIL
import cv2
import numpy as np
from PIL import Image


def load_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))

def extract_cropped_faces_from_video(filename_video, path_to_boxes, path_to_save_faces='', new_size=(224,224)):
    # TODO: Во всех случаях нет некоторых файлов .pts, нужно обрабатывать такие случаи (вычеркивать их из лейблов)
    # TODO: Думаю, также нужно записывать таймстепы...чтобы потом было возможно смешать данные с другими базами
    # TODO: ПРоверь, сколько всего кадров в видео и сколько лэйблов
    cap = cv2.VideoCapture(filename_video)
    frame_rate = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS)
    total_frames = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    format_video=filename_video.split('.')[-1]
    while (cap.isOpened()):
        ret, frame = cap.read()
        if format_video=='mp4':
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if ret == False:
            break
        pts=load_pts(path_to_boxes+str(i)+'.pts')
        x,x_plus,y,y_plus=int(pts[0,0]), int(pts[2,0]), int(pts[0,1]), int(pts[2,1])
        img=Image.fromarray(frame)
        img=img.crop((x,y,x_plus,y_plus))
        img=img.resize(new_size, resample=PIL.Image.BILINEAR)
        i+=1
        if i%1000==0:
            img.show()


filename_video='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\Aff_wild\\Videos\\train\\309.mp4'
path_to_boxes='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\Aff_wild\\bboxes\\train\\309\\'
new_size=(224,224)
extract_cropped_faces_from_video(filename_video,path_to_boxes,'',new_size)