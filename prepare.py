import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Пути к данным
base_dir = 'dataset'
labels_csv_path = os.path.join(base_dir, 'labels.csv')
all_images_dir = os.path.join(base_dir, 'all_images')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

# Чтение labels.csv
labels_df = pd.read_csv(labels_csv_path)

# Разбиение данных на тренировочные, тестовые и валидационные
train_df, temp_df = train_test_split(labels_df, test_size=0.2,
                                     stratify=labels_df['breed'])  # 80% для тренировки, 20% временный
validation_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[
    'breed'])  # Разделение временного: 50% в тест, 50% в валидацию


# Функция для создания папок и копирования изображений
def distribute_images(df, target_dir):
    for _, row in df.iterrows():
        file_name = row['id'] + '.jpg'
        breed = row['breed']
        breed_dir = os.path.join(target_dir, breed)

        # Создание папки для породы, если она не существует
        if not os.path.exists(breed_dir):
            os.makedirs(breed_dir)

        # Копирование файла
        src_path = os.path.join(all_images_dir, file_name)
        dst_path = os.path.join(breed_dir, file_name)
        shutil.copy(src_path, dst_path)


# Распределение изображений
distribute_images(train_df, train_dir)
distribute_images(test_df, test_dir)
distribute_images(validation_df, validation_dir)

# Подсчет количества изображений в каждой папке
def count_images_per_class(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            count = len(os.listdir(class_dir))
            class_counts[class_name] = count
    return class_counts

train_counts = count_images_per_class(train_dir)
test_counts = count_images_per_class(test_dir)
validation_counts = count_images_per_class(validation_dir)

import numpy as np

labels_df = pd.read_csv(labels_csv_path)

# Выбор 60 классов
selected_breeds = np.random.choice(labels_df['breed'].unique(), 60, replace=False)
labels_df = labels_df[labels_df['breed'].isin(selected_breeds)]

# Удаление содержимого папок
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

clear_directory(train_dir)
clear_directory(test_dir)
clear_directory(validation_dir)

# Определение минимального количества изображений на класс
min_count = labels_df['breed'].value_counts().min()

# Фильтрация каждого класса до min_count изображений
balanced_df = pd.DataFrame()
for breed in selected_breeds:
    breed_df = labels_df[labels_df['breed'] == breed].sample(min_count, random_state=42)
    balanced_df = pd.concat([balanced_df, breed_df])

# Разделение и распределение изображений
train_df, temp_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['breed'])
validation_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['breed'])

distribute_images(train_df, train_dir)
distribute_images(test_df, test_dir)
distribute_images(validation_df, validation_dir)


def save_breeds(directory, save_path):
    """
    Сохраняет список названий пород, найденных в указанной директории, в текстовый файл.

    """
    # Получаем список поддиректорий в указанной директории
    breeds = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    breeds.sort()  # Опционально, сортировка списка

    # Сохраняем список пород в файл
    with open(save_path, 'w') as file:
        for breed in breeds:
            file.write(breed + '\n')


# Использование функции
# Путь к тренировочной папке
save_path = os.path.join(base_dir, 'breeds_list.txt')  # Путь к файлу для сохранения списка пород
save_breeds(train_dir, save_path)