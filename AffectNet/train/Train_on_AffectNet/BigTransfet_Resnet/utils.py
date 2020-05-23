import numpy as np
import pandas as pd
from PIL import Image

def load_image(path=''):
    img = Image.open(path)
    img = img.convert('RGB')
    img = np.array(img)  # image has been transposed into (height, width)
    return img

def normalize_image(image):
    image=image/255.
    return image

def load_normalize_image(path=''):
    image=load_image(path)
    image=normalize_image(image)
    return image

def load_FAU(path=''):
    FAU=pd.read_csv(path)
    return FAU.iloc[0,2:19].values