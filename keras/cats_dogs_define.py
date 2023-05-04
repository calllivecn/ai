#!/usr/bin/env python3
# coding=utf-8
# date 2019-01-24 15:02:35
# https://github.com/calllivecn

"""
dogs_vs_cats = https://www.kaggle.com/c/dogs-vs-cats/data  下载数据(需要kaggle账号)

"""

__all__ = ["base_dir", "train_dir", "validation_dir", "test_dir",
            "train_cats_dir", "train_dogs_dir", "validation_cats_dir",
            "validation_dogs_dir", "test_cats_dir", "test_dogs_dir"]

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

