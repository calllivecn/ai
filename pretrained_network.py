#!/usr/bin/env python3
# coding=utf-8
# date 2019-01-23 14:42:26
# https://github.com/calllivecn


# 将VGG16卷积基实例化
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150, 150, 3))


conv_base.summary()
