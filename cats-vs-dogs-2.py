#!/usr/bin/env py3
#coding=utf-8
# date 2019-01-22 22:52:55
# author calllivecn <c-all@qq.com>

"""
dogs_vs_cats = https://www.kaggle.com/c/dogs-vs-cats/data  下载数据(需要kaggle账号)

"""

import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

# 代码清单5-4，将图像复制到训练、验证和测试的目录。(整理好后定义下列变量)

import os

origin_dataset_dir = "/home/zx/.keras/dog-cat"

base_dir = os.path.join(origin_dataset_dir, "cats_and_dogs_small")

train_dir = os.path.join(base_dir, "train")

validation_dir = os.path.join(base_dir, "validation")

test_dir = os.path.join(base_dir, "test")

train_cats_dir = os.path.join(train_dir, 'cats')

train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, "cats")

validation_dogs_dir = os.path.join(validation_dir, "dogs")

test_cats_dir = os.path.join(test_dir, 'cats')

test_dogs_dir = os.path.join(test_dir, 'dogs')

# 代码清单5-11，利用ImageDataGenerator来设置数据增强


datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

# 代码清单5-12，显示几个随机增强的训练图像
    # 略。。。

# 代码清单5-13，定义一个包含dropout的新卷积神经网络


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc']
                )


# 代码清单5-14，利用数据增强生成器训练卷积神经网络

train_datagen = ImageDataGenerator(rescale=1.0/255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
                                                train_dir, # 目录
                                                target_size=(150, 150), # 奖所有图像的大小调整为150x150
                                                batch_size=32,
                                                class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                                                validation_dir,
                                                target_size=(150, 150),
                                                batch_size=32,
                                                class_mode='binary')


history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50)

# 代码清单5-15，保存模型

model.save('cats_and_dogs_small_2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) +1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
