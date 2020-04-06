import PIL
import numpy as np
import AffectNet.train.VGGface2.src.config as cg

def load_preprocess_image(path=''):
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    x = np.array(img)  # image has been transposed into (height, width)
    x = x[:, :, ::-1] - cg.mean
    return x

