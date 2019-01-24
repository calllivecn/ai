#!/usr/bin/env python3
# coding=utf-8
# date 2019-01-24 14:52:44
# https://github.com/calllivecn

from cats_dogs_define import *

import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.preprocessing import image


model = models.load_model('cats_and_dogs_small_2.h5')

model.summary()

# 5-25

img_path = os.path.join(test_cats_dir, "cat.1700.jpg")
print(img_path)

img = image.load_img(img_path, target_size=(150, 150))

img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.0 # 记住，训练模型的输入数据都用这种方法预处理

# 其形状为(1, 150, 150, 3)
print(img_tensor.shape)


# 5-26，显示测试图像

plt.imshow(img_tensor[0])
plt.show()


# 5-27，用一个输入张量和一个输出张量列表将模型实例化

layer_outputs = [ layer.output for layer in model.layers[:8] ] # 提取前8层的输出
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 5-28，以预测模式运行模型

activations = activation_model.predict(img_tensor) # 返回8个Numpy数组组成的列表，每个层激活对应一个Numpy数组
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# 5-[29，30]将第4个通道可视化

def visual(f_l_a, num):
    plt.matshow(f_l_a[0, :, :, num], cmap='viridis')
    plt.show()

#visual(first_layer_activation, 4)
#visual(first_layer_activation, 7)

# 5-31，将每个中间激活的所有通道可视化

# 层的名称，这样你可以将这些名称画到图中
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
# 层的名称，这样你可以将这些名称画到图中

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # 显示特征图
    n_features = layer_activation.shape[-1] # 特征图中的特征个数

    size = layer_activation.shape[1] # 特征图的形状为(1, size, size, n_features)

    n_cols = n_features // images_per_row # 在这个矩阵中将激活通道平铺
    display_grid = np.zeros((size * n_cols, images_per_row * size))


    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= (channel_image.std() + 1e-5)
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1.0 / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()





