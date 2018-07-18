#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
import numpy as np

import sys
from PIL import Image
sys.modules['Image'] = Image

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

from keras.applications.resnet50 import preprocess_input, decode_predictions

classes_dict = {0: '闭眼',
                1: '抽烟',
                2: '打手机',
                3: '打哈欠',
                4: '左顾右盼'}

img_path = './data/test3.jpg'
img = image.load_img(img_path, target_size=(240, 360))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = load_model('./models/resnet50-imagenet-finetune152-adam.h5')
pres = model.predict(x)

# 这是一个numpy
data_pre = pres[0]
print('data_pre---->', data_pre)
confidence = np.max(data_pre)
print('最大confidence---->', confidence)

# numpy array ----> list
data_pre_list = data_pre.tolist()

index = data_pre_list.index(max(data_pre_list))
# print('最大值索引----->', index)
print("最大值为：{}，属于--{}".format(confidence, classes_dict[index]))



