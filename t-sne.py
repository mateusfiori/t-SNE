import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

comments = pd.read_csv("./comments.csv").values

#Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comments.ravel())

X_embedded = TSNE(n_components=3).fit_transform(X.toarray())

print(X_embedded[1])