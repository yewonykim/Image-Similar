import os
import numpy as np
from PIL import Image

from model.FeatureExtractor import FeatureExtractor

__all__ = ["process_images"]

def process_images(directory):
    fe = FeatureExtractor()
    features = []
    img_paths = []

    img_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    for img_name in img_files:
        try:
            image_path = os.path.join(directory, img_name)
            img_paths.append(image_path)
            
            feature = fe.extract(img=Image.open(image_path))
            features.append(feature)
            
            feature_path = "./" + os.path.splitext(img_name)[0] + ".npy"
            np.save(feature_path, feature)
        except Exception as e:
            print('예외가 발생했습니다.', e)

    return fe, features, img_paths