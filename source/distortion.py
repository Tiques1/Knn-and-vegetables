import cv2
import numpy as np
import random
import os

class Distortion:
    @staticmethod
    def rotate_image(image, angle):
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    @staticmethod
    def flip_image(image, flip_code):
        flipped_image = cv2.flip(image, flip_code)
        return flipped_image

    # Сдвиг
    @staticmethod
    def shift_image(image, shift_x, shift_y):
        height, width = image.shape[:2]
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_image = cv2.warpAffine(image, translation_matrix, (width, height))
        return shifted_image

    @staticmethod
    def adjust_brightness(image, brightness_factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
        brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return brightened_image

    @staticmethod
    def add_noise(image, mean=0, std=10):
        noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image







# Функция для применения всех искажений к изображению с рандомными параметрами
def apply_distortions(image_path, output_folder):
    image = cv2.imread(image_path)

    # Генерируем случайные параметры для каждой функции
    angle = random.randint(-30, 30)
    flip_code = random.randint(-1, 1)  # -1: по вертикали, 0: не флип, 1: по горизонтали
    shift_x = random.randint(-50, 50)
    shift_y = random.randint(-50, 50)
    brightness_factor = random.uniform(0.5, 2.0)
    noise_mean = 0
    noise_std = random.randint(5, 20)

    # Применяем каждую функцию
    # rotated_image = Distortion.rotate_image(image, angle)
    # flipped_image = Distortion.flip_image(image, flip_code)
    # shifted_image = Distortion.shift_image(image, shift_x, shift_y)
    # brightened_image = Distortion.adjust_brightness(image, brightness_factor)
    noisy_image = Distortion.add_noise(image, noise_mean, noise_std)

    # Сохраняем изображение
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_name)
    print(image_name, output_path)
    c = 9000
    while os.path.exists(output_path):
        image_name = os.path.splitext(os.path.basename(image_path))[0] + str(c) + '.jpg'
        output_path = os.path.join(output_folder, image_name)
        c += 1
    # cv2.imwrite(output_path, noisy_image)
    # cv2.imwrite(output_path, flipped_image)
    # cv2.imwrite(output_path, shifted_image)
    # cv2.imwrite(output_path, brightened_image)
    cv2.imwrite(output_path, noisy_image)


# Папка с исходными изображениями
input_folder = "../vegetables/"
# Папка для сохранения обработанных изображений
output_folder = "../vegetables_dist/"

# Создаем папку для сохранения, если она не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Применяем функцию к каждому изображению в папке
for image_file in os.listdir(input_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(input_folder, image_file)
        apply_distortions(image_path, output_folder)