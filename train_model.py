import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Пути
processed_data_path = './processed_data.pkl'
model_path = './license_plate_model.h5'

# Параметры модели
input_shape = (64, 128, 1)
max_seq_len = 8
num_classes = 36  # 26 букв + 10 цифр

# Загрузка обработанных данных
with open(processed_data_path, 'rb') as f:
    images, encoded_labels = pickle.load(f)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Создание генератора для увеличения данных
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Применение увеличения данных
datagen.fit(X_train)

# Построение модели
def build_model(input_shape, num_classes, max_seq_len):
    model = models.Sequential()

    # CNN-слои
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # Дополнительный сверточный слой
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))  # Еще один сверточный слой
    model.add(layers.MaxPooling2D((2, 2)))

    # Извлечение признаков
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Исправленный слой для создания временных шагов
    model.add(layers.RepeatVector(max_seq_len))  # Создаем временные шаги
    model.add(layers.LSTM(128, return_sequences=True))

    # Выходной слой
    model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='softmax')))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Обучение модели с использованием генератора данных
print("[INFO] Обучение модели...")
model = build_model(input_shape, num_classes, max_seq_len)
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=80, validation_data=(X_test, y_test))
model.save(model_path)
print(f"[INFO] Модель сохранена в {model_path}.")
