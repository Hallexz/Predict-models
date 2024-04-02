from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


class mypca:
    def __init__(self, n_components=10):
        self.pca = PCA(n_components=n_components)
        self.similar_columns = []
        self.accuracy = None
        
    def fit(self, df):
        self.pca.fit(df)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        
        importance = self.pca.components_
        for i in range(len(importance[0])):
            for j in range(i+1, len(importance[0])):
                if np.isclose(importance[0][i], importance[0][j]):
                    self.similar_columns.append((df.columns[i], df.columns[j]))
        return self.similar_columns if self.similar_columns else []

    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio_

class mykmeans:
    def __init__(self, n_clusters=None, random_state=0):
        self.kmeans = KMeans(n_clusters=n_clusters if n_clusters is not None else 8, random_state=random_state)
        self.labels_ = None
        self.silhouette_score_ = None

    def fit(self, df):
        self.kmeans.fit(df)
        self.labels_ = self.kmeans.labels_
        self.silhouette_score_ = silhouette_score(df, self.labels_)
        return self.labels_

    def get_silhouette_score(self):
        return self.silhouette_score_

    def merge_similar_columns(self, df, similar_columns):
        for col1, col2 in similar_columns:
            df[col1] = df[col1] + df[col2].median(axis=1)
            df = df.drop(columns=col2)
        return df