import numpy as np
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