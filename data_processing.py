import os
import cv2
import json
import numpy as np
import pickle
import string
from tensorflow.keras.preprocessing.image import img_to_array

# Алфавит и вспомогательные функции
alphabet = string.ascii_uppercase + string.digits
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}

# Пути
image_dir = './data/train/img'
annotations_dir = './data/train/ann/'
processed_data_path = './processed_data.pkl'
max_seq_len = 8  # Максимальная длина номера

# Функция для кодирования меток
def encode_labels(labels, max_seq_len):
    encoded_labels = []
    for label in labels:
        encoded_label = [char_to_idx.get(char, 0) for char in label]
        encoded_label = encoded_label[:max_seq_len] + [0] * (max_seq_len - len(encoded_label))
        encoded_labels.append(encoded_label)
    return np.array(encoded_labels)

# Функция для загрузки изображений и меток
def load_images_and_labels(image_dir, annotations_dir):
    images = []
    labels = []

    for image_filename in os.listdir(image_dir):
        image_name = os.path.splitext(image_filename)[0]

        image_path = os.path.join(image_dir, image_filename)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 64))
        img = img_to_array(img) / 255.0
        images.append(img)

        annotation_path = os.path.join(annotations_dir, image_name + '.json')
        with open(annotation_path) as f:
            annotation = json.load(f)

        labels.append(annotation['description'])

    return np.array(images), labels

# Основной код
if not os.path.exists(processed_data_path):
    print("[INFO] Загружаем и обрабатываем данные...")
    images, labels = load_images_and_labels(image_dir, annotations_dir)
    encoded_labels = encode_labels(labels, max_seq_len)

    with open(processed_data_path, 'wb') as f:
        pickle.dump((images, encoded_labels), f)
    print("[INFO] Данные успешно обработаны и сохранены.")
else:
    print("[INFO] Обработанные данные уже существуют.")
