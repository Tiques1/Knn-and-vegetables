import cv2
import os


class VidToImgs:
    @staticmethod
    def extract_frames(video, output_folder, counter):
        video = cv2.VideoCapture(video)

        if not video.isOpened():
            print("Video opening error")
            return

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Counter is number which concatenate to name (for copies). Becareful to won't overwrite already exists img
        frame_count = counter
        while True:
            success, frame = video.read()

            # If we can't read, video is over
            if not success:
                break

            # Save image
            frame_path = os.path.join(output_folder, f"{os.path.basename(video)}{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_count += 1
        video.release()


if __name__ == '__main__':
    vti = VidToImgs()

    for i in ['Mandarin', 'Ogurec', 'Yabloko', 'Pomidor']:

        video_path = f"D:\\Vegetables\\{i}\\{i}.mp4"
        output_folder = f"D:\\Vegetables\\{i}\\"

        vti.extract_frames(video_path, output_folder, 10000)
