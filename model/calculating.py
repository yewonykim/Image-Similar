from pathlib import Path
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_similarity(fe, features, img_paths, query_image_path):
    img = Image.open(query_image_path)
    query = fe.extract(img)
    dists = np.linalg.norm(features - query, axis=1)
    ids = np.argsort(dists)[:30]
    scores = [(dists[id], img_paths[id]) for id in ids]

    axes = []
    fig = plt.figure(figsize=(8, 8))
    for a in range(5 * 6):
        score = scores[a]
        axes.append(fig.add_subplot(5, 6, a + 1))
        subplot_title = str(score[0])
        axes[-1].set_title(subplot_title)
        plt.axis('off')
        plt.imshow(Image.open(score[1]))

    fig.tight_layout()
    plt.show()