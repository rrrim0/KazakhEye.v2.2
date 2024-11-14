import os
import cv2
import numpy as np
import string
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

# Пути
model_path = './license_plate_model.h5'
processed_data_path = './processed_data.pkl'

# Алфавит
alphabet = string.ascii_uppercase + string.digits
idx_to_char = {idx: char for idx, char in enumerate(alphabet)}

# Словарь регионов Казахстана
regions = {
    "01": "г. Нур-Султан (Астана)",
    "02": "г. Алматы",
    "03": "Акмолинская область",
    "04": "Актюбинская область",
    "05": "Алматинская область",
    "06": "Атырауская область",
    "07": "Западно-Казахстанская область",
    "08": "Жамбылская область",
    "09": "Карагандинская область",
    "10": "Костанайская область",
    "11": "Кызылординская область",
    "12": "Мангистауская область",
    "13": "Туркестанская область",
    "14": "Павлодарская область",
    "15": "Северо-Казахстанская область",
    "16": "Восточно-Казахстанская область",
    "17": "г. Шымкент",
    "18": "Абайская область",
    "19": "Жетысуская область",
    "20": "Улытауская область",
}

# Загрузка модели
print("[INFO] Загружаем модель...")
model = load_model(model_path)

# Функция для обработки предсказанного номера
def process_predicted_number(predicted):
    # Удаляем префикс "kz" или "KZ", если он есть
    if predicted.lower().startswith("kz"):
        predicted = predicted[2:]

    # Проверяем длину предсказанного номера
    if len(predicted) < 8:
        return f"Предсказанный номер: {predicted}\nРегион: Регион не обнаружен"

    # Извлекаем последние две цифры номера
    region_code = predicted[-2:]

    # Проверяем код региона в таблице
    region_name = regions.get(region_code, "Регион не обнаружен")

    # Возвращаем результат
    return f"Предсказанный номер: {predicted}\nРегион: {region_code} - {region_name}"

# Функция для предсказания
def predict_license_plate(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_text = ''.join([idx_to_char[np.argmax(pred)] for pred in prediction[0]])
    return predicted_text

# Создание графического интерфейса
def create_interface():


    def convert_image_to_png(image_path):
        """
        Конвертирует изображение в формат PNG, если оно не в PNG.
        """
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in ['.png']:  # Если файл не в формате PNG
            # Загружаем изображение
            img = Image.open(image_path)
            # Конвертируем в PNG
            png_image_path = os.path.splitext(image_path)[0] + '.png'
            img.save(png_image_path, 'PNG')
            return png_image_path
        return image_path

    def load_image(image_path):
        """
        Загружает изображение, конвертируя его в PNG, если это не PNG.
        """
        image_path = convert_image_to_png(image_path)
        # Теперь image_path точно будет указывать на PNG файл
        image = Image.open(image_path)
        image = image.resize((300, 150), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(image)
        label_image.config(image=img)
        label_image.image = img
        entry_image_path.delete(0, tk.END)
        entry_image_path.insert(0, image_path)

    def recognize_plate():
        image_path = entry_image_path.get()
        if not image_path:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return

        try:
            predicted_text = predict_license_plate(image_path, model)
            result = process_predicted_number(predicted_text)
            label_result.config(text=result)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось распознать номер: {e}")

    def clear_fields():
        label_result.config(text="")
        label_image.config(image='')
        label_image.image = None
        entry_image_path.delete(0, tk.END)

    # Создаем окно
    root = tk.Tk()
    root.title("Распознавание номера автомобиля")

    # Настройка шрифта
    font = ("Montserrat", 12)

    # Кнопка загрузки изображения
    button_load_image = tk.Button(root, text="Загрузить", font=font, command=load_image)
    button_load_image.grid(row=0, column=0, padx=10, pady=10)

    # Поле для отображения пути изображения
    entry_image_path = tk.Entry(root, font=font, width=40)
    entry_image_path.grid(row=0, column=1, padx=10, pady=10)

    # Рамка для отображения изображения
    label_image = tk.Label(root)
    label_image.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    # Кнопка для распознавания номера
    button_recognize = tk.Button(root, text="Распознать", font=font, command=recognize_plate)
    button_recognize.grid(row=2, column=0, padx=10, pady=10)

    # Кнопка очистки
    button_clear = tk.Button(root, text="Очистить", font=font, command=clear_fields)
    button_clear.grid(row=2, column=1, padx=10, pady=10)

    # Метка для отображения результата
    label_result = tk.Label(root, text="Результат", font=font, justify="left")
    label_result.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    # Запуск интерфейса
    root.mainloop()

if __name__ == "__main__":
    create_interface()
