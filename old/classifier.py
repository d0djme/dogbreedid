import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Пути к данным
base_dir = 'C:/Users/Тимур/Desktop/study/summer_practic/dog-breed-identification'  # Путь к директории с данными
labels_csv_path = os.path.join(base_dir, 'labels.csv')
all_images_dir = os.path.join(base_dir, 'all_images')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Чтение labels.csv
labels_df = pd.read_csv(labels_csv_path)

# Разбиение данных на тренировочные и тестовые
train_df, test_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['breed'])

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

print("Изображения распределены по папкам.")
