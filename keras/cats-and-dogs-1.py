#!/usr/bin/env py3
#coding=utf-8
# date 2019-01-22 22:52:55
# author calllivecn <calllivecn@outlook.com>

"""
dogs_vs_cats = https://www.kaggle.com/c/dogs-vs-cats/data  下载数据(需要kaggle账号)

"""


# 代码清单5-4，将图像复制到训练、验证和测试的目录。(整理好后定义下列变量)

from cats_dogs_define import *

# 代码清单5-5，将猫狗分类的小型卷积神经网络实例化

from keras import layers, models


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

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


# 代码清单5-6，配置模型用于训练

from keras import optimizers

model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc']
                )

# 数据预处理
"""
(1) 读取图像文件。
(2) 将JPEG文件解码为RGB像素网格。
(3) 将这些像素风格转换为浮点数张量。
(4) 将像素值(0～255范围内)缩放到[0, 1]区间(正如你所知，神经网络喜欢处理较小的输入值)。
"""

# 代码清单5-7，使用ImageDataGenerator从目录中读取图像

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255) # 将所有图像缩放
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
                                                train_dir, # 目录
                                                target_size=(150, 150), # 奖所有图像的大小调整为150x150
                                                batch_size=20,
                                                class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                                                validation_dir,
                                                target_size=(150, 150),
                                                batch_size=20,
                                                class_mode='binary')

# 代码清单5-8 利用批量生成器拟合模型

history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50)

# 代码清单5-9，始终在训练完成后保存模型，这是一种良好的实践。

model.save('cats_and_dogs_small_1.h5')


# 代码清单5-10，绘制训练过程中的损失曲线和精度曲线。

import matplotlib.pyplot as plt

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
