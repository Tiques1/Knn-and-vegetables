import threading
import cv2
import joblib
import matplotlib.pyplot as plt


class RealtimeKNN:
    def __init__(self):
        self.frame = None
        self.cam = None

    def camera_frame(self, camera='0', display=True):
        self.cam = cv2.VideoCapture(camera)
        if not self.cam.isOpened():
            print("Ошибка при открытии камеры")
            return

        while True:
            ret, my_frame = self.cam.read()
            if not ret:
                print("Ошибка при чтении кадра")
                return

            self.frame = my_frame

            if display is True:
                cv2.imshow('Video', cv2.transpose(my_frame))

            if cv2.waitKey(1) == 'q':
                break

        self.cam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def preprocessing(my_frame, my_scaler, my_pca):
        try:
            new_size = (75, 40)
            resized_image = cv2.resize(cv2.transpose(my_frame), new_size, interpolation=cv2.INTER_AREA)

            flattened_image = resized_image.flatten()

            scaled = my_scaler.transform([flattened_image])

            scaled_pca = my_pca.transform(scaled)
            return scaled_pca
        except Exception as e:
            pass

    @staticmethod
    # The model is one of knn or kmeans
    def predict(model, prepocessed):
        try:
            result = model.predict(prepocessed)
            return result
        except Exception as e:
            pass

    @staticmethod
    def graph():
        my_fig, my_ax = plt.subplots()

        my_ax.set_xlabel('PCA Component 1')
        my_ax.set_ylabel('PCA Component 2')
        my_ax.set_title('KMeans Clustering')
        return my_fig, my_ax

    @staticmethod
    def graph_update(my_fig, my_ax, cluster, scaled_pca):
        colors = {'Pomidor': 'red', 'Ogurec': 'green', 'Mandarin': 'orange', None: 'black'}
        color = colors.get(cluster[0].astype(str))

        my_ax.scatter(scaled_pca[0, 0], scaled_pca[0, 1], c=color, s=100, marker='o')

        # Redrawing the graph and processing the events
        my_fig.canvas.draw()
        my_fig.canvas.flush_events()

        # Pause to display
        plt.pause(0.01)


if __name__ == "__main__":
    knn = joblib.load('knn40k.pkl')
    scaler = joblib.load('scaler40k.pkl')
    pca = joblib.load('pca40k.pkl')

    rltm = RealtimeKNN()
    # fig, ax = rltm.graph()

    threading.Thread(target=rltm.camera_frame, args=[f"http://192.168.1.50:9999/video"]).start()
    while True:
        frame = rltm.preprocessing(rltm.frame, scaler, pca)
        prediction = rltm.predict(knn, frame)
        # rltm.graph_update(fig, ax, prediction, frame)
        print(prediction)
