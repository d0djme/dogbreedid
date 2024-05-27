import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'dataset')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
model_dir = os.path.join(base_dir, 'models')
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
# Загрузка модели
baest = os.path.join(base_dir, 'best_model.keras')
model = load_model(baest)
results = []


def load_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def predict(img_path):
    img_array = load_image(img_path)
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    return predicted_class


# Список пород собак
def load_breeds(file_path):
    with open(file_path, 'r') as file:
        breeds = file.read().splitlines()
    return breeds


# Загрузка списка пород
breeds_file_path = os.path.join(data_dir, 'breeds_list.txt')  # Путь к файлу со списком пород
breeds = load_breeds(breeds_file_path)


def open_file():
    filepath = filedialog.askopenfilename(
        title="Open file",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not filepath:
        return

    correct_breed = os.path.basename(os.path.dirname(filepath))
    predicted_class = predict(filepath)
    predicted_breed = breeds[predicted_class]

    if predicted_breed == correct_breed:
        color = 'green'
    else:
        color = 'red'

    result_text = f"Predicted breed: {predicted_breed}\nCorrect breed: {correct_breed}"
    label.config(text=result_text, fg=color)

    img = Image.open(filepath)
    img = img.resize((250, 250), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # Сохраняем результаты для Jupyter
    results.append((filepath, result_text))


# Создание GUI
def load_display_image(img_path):
    img = Image.open(img_path)
    img = img.resize((250, 250), Image.LANCZOS)  # Изменяем размер для отображения
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Сохраняем ссылку на изображение


root = tk.Tk()
root.title("Dog Breed Predictor")
root.geometry("500x600")  # Размер окна

btn_open = tk.Button(root, text="Open Image", command=open_file)
btn_open.pack(pady=20)

label = tk.Label(root, text="Select an image")
label.pack(pady=20)

image_label = tk.Label(root)  # Метка для отображения изображения
image_label.pack(pady=20)

root.mainloop()