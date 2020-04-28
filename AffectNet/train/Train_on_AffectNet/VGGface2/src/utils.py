import numpy as np
from PIL import Image

import AffectNet.train.Train_on_AffectNet.VGGface2.src.config as cg

def load_image(path=''):
    img = Image.open(path)
    img = img.convert('RGB')
    img = np.array(img)  # image has been transposed into (height, width)
    return img

def preprocess_image(image):
    x = image[:, :, ::-1] - cg.mean
    return x

def load_preprocess_image(path=''):
    image=load_preprocess_image(path)
    image=preprocess_image(image)
    return image