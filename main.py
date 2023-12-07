# Import the libraries
%matplotlib inline

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import os
class FeatureExtractor:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
fe = FeatureExtractor()
features = []
img_paths = []

# 이미지가 있는 폴더 경로
directory = "./pokemon/images/images"
# 폴더 내 모든 파일 이름을 가져옵니다.
img_files = [f for f in os.listdir(directory) if f.endswith('.png')]
for img_name in img_files:
    try:
        image_path = os.path.join(directory, img_name)
        img_paths.append(image_path)
        # Extract Features
        feature = fe.extract(img=Image.open(image_path))
        features.append(feature)
        # Save the Numpy array (.npy) on designated path
        feature_path = "./feature/" + os.path.splitext(img_name)[0] + ".npy"
        np.save(feature_path, feature)
    except Exception as e:
        print('예외가 발생했습니다.', e)
# Import the libraries
import matplotlib.pyplot as plt
import numpy as np
# Insert the image query
img = Image.open("./pokemon/images/images/{NAME}.png")
# Extract its features
query = fe.extract(img)
# Calculate the similarity (distance) between images
dists = np.linalg.norm(features - query, axis=1)
# Extract 30 images that have lowest distance
ids = np.argsort(dists)[:30]
scores = [(dists[id], img_paths[id]) for id in ids]
# Visualize the result
axes=[]
fig=plt.figure(figsize=(8,8))
for a in range(5*6):
    score = scores[a]
    axes.append(fig.add_subplot(5, 6, a+1))
    subplot_title=str(score[0])
    axes[-1].set_title(subplot_title)  
    plt.axis('off')
    plt.imshow(Image.open(score[1]))
fig.tight_layout()
plt.show()