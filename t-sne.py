import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

dataset_size = 10

#shape -> [10000, 1]
#comments = pd.read_csv("./comments.csv").values
comments_neg = pd.read_csv("./neg.csv").values
comments_pos = pd.read_csv("./pos.csv").values

comments = np.concatenate((comments_neg[0:dataset_size], comments_pos[0:dataset_size]))  # original size was taking too long to process

print("Waiting for BOW")
#Bag of Words
vectorizer = CountVectorizer(stop_words='english', lowercase=True, max_df=0.95, min_df=0.01)
X = vectorizer.fit_transform(comments.ravel())
print(X.shape)
print("BOW done...\n")

print("Waiting for TSVD")
#TruncatedSVD - dimensionality reduction for sparse data
tSVD = TruncatedSVD(n_components=20)
dec_x = tSVD.fit_transform(X)
print("TSVD done...\n")

print("Waiting for T-SNE...")
#t-SNE
X_embedded = TSNE(n_components=3, verbose=1, learning_rate=300.0).fit_transform(dec_x)
print("T-SNE done...\n")

classes = []

for i in np.arange(0, dataset_size*2):
    if i < 10:
        classes.append(0)
    else:
        classes.append(1)

dictColor = {
    0.: 'r',
    1.: 'b'
}

X_embedded = np.append(X_embedded, np.asarray(classes)[:, None], axis=1)

print(X_embedded.shape)

#plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for plot in X_embedded:
    ax.scatter(plot[0], plot[1],
               plot[2], c=dictColor[plot[3]])

plt.show()
