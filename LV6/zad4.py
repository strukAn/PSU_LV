import scipy as sp
from sklearn import cluster, datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    face = plt.imread('example_grayscale.png')
except AttributeError:
    from scipy import misc
    face = misc.face(gray=True)
    
X = face.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=8,n_init=1)
k_means.fit(X) 
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

plt.figure(1)
plt.imshow(face,  cmap='gray')
plt.show()

plt.figure(2)
plt.imshow(face_compressed,  cmap='gray')
plt.show()

print(face.size * 8) 

# roundUp(ld(10)) = 4
print(face_compressed.size * 4)
print('2 : 1')