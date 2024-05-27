import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Путь к директории с тестовыми данными
test_dir = 'C:/Users/Тимур/Desktop/study/summer_practic/dog-breed-identification/test'

# Создание объекта ImageDataGenerator для тестового набора данных
test_datagen = ImageDataGenerator(rescale=1./255)

# Создание генератора данных для тестовых изображений
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Загрузка обученной модели
model = load_model('C:/Users/Тимур/best_model.keras')
# Оценка модели на тестовом наборе данных
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss}")
