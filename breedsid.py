import os

def save_breeds(directory, save_path):
    """
    Сохраняет список названий пород, найденных в указанной директории, в текстовый файл.
    
    Args:
    directory (str): Путь к директории с поддиректориями, названными по породам собак.
    save_path (str): Путь к файлу, куда будет сохранён список пород.
    """
    # Получаем список поддиректорий в указанной директории
    breeds = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    breeds.sort()  # Опционально, сортировка списка
    
    # Сохраняем список пород в файл
    with open(save_path, 'w') as file:
        for breed in breeds:
            file.write(breed + '\n')

# Использование функции
directory_path = 'C:/Users/Тимур/Desktop/study/summer_practic/dog-breed-identification/train'  # Путь к тренировочной папке
save_path = 'C:/Users/Тимур/Desktop/study/summer_practic/breeds_list.txt'  # Путь к файлу для сохранения списка пород
save_breeds(directory_path, save_path)
