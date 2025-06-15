import scipy as sp
from sklearn import cluster, datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    face = plt.imread('example.png')
except AttributeError:
    from scipy import misc
    face = misc.face(gray=False)
    
X = face.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=4,n_init=1)
k_means.fit(X) 
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

plt.figure(1)
plt.imshow(face)
plt.show()

plt.figure(2)
plt.imshow(face_compressed)
plt.show()