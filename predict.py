import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Загрузка модели
model = load_model('C:/Users/Тимур/best_model.keras') #нужен путь к модели

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
breeds_file_path = 'C:/Users/Тимур/Desktop/study/summer_practic/breeds_list.txt'  # Путь к файлу со списком пород
breeds = load_breeds(breeds_file_path)


def open_file():
    filepath = filedialog.askopenfilename(
        title="Open file",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not filepath:
        return
    predicted_class = predict(filepath)
    breed = breeds[predicted_class]
    label.config(text=f"Predicted breed: {breed}")

    # Загружаем и отображаем изображение
    load_display_image(filepath)

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
