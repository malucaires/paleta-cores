import io
from PIL import Image
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.cluster import KMeans


def get(uploaded_image, n_clusters):
    with open(uploaded_image.name,'wb') as f:
        f.write(uploaded_image.getbuffer())
    image = Image.open(uploaded_image.name)
    image.thumbnail((256,256), Image.Resampling.LANCZOS)
    N, M = image.size
    x = np.asarray(image).reshape((N*M),3)
    model = KMeans(n_clusters, random_state=42).fit(x)
    cores = model.cluster_centers_.astype('uint8')[np.newaxis]
    cores_hex = [matplotlib.colors.to_hex(cor/255) for cor in cores[0]]
    Path(uploaded_image.name).unlink()
    return cores, cores_hex

                        
def show(cores):
    fig = plt.figure()
    plt.imshow(cores)
    plt.axis('off')
    return fig


def save (fig):
    img = io.BytesIO()
    plt.savefig(img, format = 'png')
    return img
