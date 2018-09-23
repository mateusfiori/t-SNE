import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

#shape -> [10000, 1]
comments = pd.read_csv("./comments.csv").values
comments = comments[0:1000] #original size was taking too long to process


print("Waiting for BOW")
#Bag of Words
vectorizer = CountVectorizer(stop_words='english', lowercase=True, max_df=0.95, min_df=0.05)
X = vectorizer.fit_transform(comments.ravel())
print("BOW done...\n")

print("Waiting for TSVD")
#TruncatedSVD - dimensionality reduction for sparse data
tSVD = TruncatedSVD(n_components=50)
dec_x = tSVD.fit_transform(X)
print("TSVD done...\n")

print("Waiting for T-SNE...")
#t-SNE
X_embedded = TSNE(n_components=3, verbose=1).fit_transform(dec_x)
print("T-SNE done...\n")

print(X_embedded)
