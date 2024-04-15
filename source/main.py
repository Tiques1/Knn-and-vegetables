import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import datetime


class Model:
    @staticmethod
    def load(path):
        images = []
        labels = []
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                file = open(os.path.join(path, filename), "rb")
                bytes_arr = bytearray(file.read())
                numpyarray = np.asarray(bytes_arr, dtype=np.uint8)

                img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                images.append(img)

                last_dot_index = Path(filename).stem.rfind(".")
                labels.append(filename[:last_dot_index])

        for symbol in '(0123456789) ':
            labels = [label.replace(symbol, '') for label in labels]

        return images, labels

    @staticmethod
    def resize(folders: [str], size, out):
        for path in folders:
            for filename in os.listdir(path):
                if filename.endswith(".jpg"):
                    file = open(os.path.join(path, filename), "rb")
                    bytes_arr = bytearray(file.read())
                    numpyarray = np.asarray(bytes_arr, dtype=np.uint8)

                    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f'{out}\\{filename}', img)

    @staticmethod
    def to_1dim(images: [cv2.typing.MatLike]):
        return [image.flatten() for image in images]

    @staticmethod
    def split(images, labels):
        x = np.array(images)
        y = np.array(labels)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def scale(x_train, x_test, dump=None):
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        if dump is not None:
            joblib.dump(scaler, f'{dump}.pkl')
        return x_train_scaled, x_test_scaled

    @staticmethod
    def pca(x_train_scaled, x_test_scaled, dump=None):
        pca = PCA(n_components=0.9, random_state=30)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)
        if dump is not None:
            joblib.dump(pca, f'{dump}.pkl')
        return x_train_pca, x_test_pca

    @staticmethod
    def knn(x_train_pca, y_train, dump=None):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train_pca, y_train)
        if dump is not None:
            joblib.dump(knn, f'{dump}.pkl')
        return knn

    @staticmethod
    def accuracy(knn, x_test_pca, y_test):
        y_pred = knn.predict(x_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    @staticmethod
    def kmeans(x_train_pca):
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(x_train_pca)
        return clusters

    @staticmethod
    def tsne_plot(data, clusters):
        tsne = TSNE(random_state=42)
        x_train_tsne = tsne.fit_transform(data)

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Clusters in t-SNE Space')

        plt.show()


if __name__ == '__main__':
    model = Model()
    images, labels = model.load('..//vegetables_dist//')

    images = model.to_1dim(images)

    x_train, x_test, y_train, y_test = model.split(images, labels)

    x_train_scaled, x_test_scaled = model.scale(x_train, x_test, 'scaler40k')

    x_train_pca, x_test_pca = model.pca(x_train_scaled, x_test_scaled, 'pca40k')

    knn = model.knn(x_train_pca, y_train, 'knn40k')
