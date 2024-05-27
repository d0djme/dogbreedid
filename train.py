import os
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import datetime

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'dataset')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
model_dir = os.path.join(base_dir, 'models')
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

batch_size = 32
img_size = 299
model_name = 'model_bs'+str(batch_size)+'_img'+str(img_size)

# Создание объектов ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Генераторы данных
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Загрузка InceptionV3 как основы
base_model = InceptionV3(weights='imagenet', include_top=False)

# Добавление новых слоев
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(60, activation='softmax')(x)  # 120 пород

# Сборка модели
model = Model(inputs=base_model.input, outputs=predictions)

# Замораживание всех слоев InceptionV3
for layer in base_model.layers:
    layer.trainable = False

# Компиляция модели
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# Callback для сохранения лучшей модели
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Подготовка генераторов данных
train_steps = len(train_df) // batch_size
validation_steps = len(validation_df) // batch_size

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,  # Используем все доступные данные
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_steps,  # Используем все доступные данные
    callbacks=[checkpoint]
)

# Размораживаем верхние слои
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# Повторная компиляция модели для дообучения
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,  # Используем все доступные данные
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_steps,  # Используем все доступные данные
    callbacks=[checkpoint]
)