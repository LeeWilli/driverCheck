#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu

import os
import cv2
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

model = load_model('./models/resnet50-imagenet-finetune152-adam.h5')

video_path = './test1'
frame_count = 1
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
write_list = []
every_write_list = []
for root, _, videos in os.walk(video_path):
    for each_video in videos:
        print("视频文件名：", each_video)
        folder_name = root.split('/')[-1]
        each_video_name = each_video.split('.')[0]
        # 得到每个视频文件路径
        each_video_full_path = os.path.join(root, each_video)
        cap = cv2.VideoCapture(each_video_full_path)
        success = True
        while (success):
            success, frame = cap.read()
            try:
                if frame_count % 50 == 1:
                    params = []
                    params.append(cv2.IMWRITE_PXM_BINARY)
                    params.append(1)
                    cv2.imwrite('test.jpg', frame, params)

                    image_path = './test.jpg'
                    img = image.load_img(image_path, target_size=(240, 360))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    print('done')

                    pres = model.predict(x)

                    # 这是一个numpy
                    data_pre = pres[0]
                    print('data_pre---->', data_pre)
                    confidence = np.max(data_pre)
                    print('最大confidence---->', confidence)
                    # 置信度大于某个阈值才认为此次分类是可靠地
                    if confidence >= 0.7:
                        # numpy array ----> list
                        data_pre_list = data_pre.tolist()

                        index = data_pre_list.index(max(data_pre_list))
                        if index == 0:
                            class1 += 1
                        elif index == 1:
                            class2 += 1
                        elif index == 2:
                            class3 += 1
                        elif index == 3:
                            class4 += 1
                        else:
                            class5 += 1
                        print("最大值为：{}，属于--{}".format(confidence, classes_dict[index]))
                        frame_count = frame_count + 1
                else:
                    frame_count = frame_count + 1
            except:
                frame_count = frame_count + 1
        cap.release()
        # 此视频完成
        # 判断哪一类读取的较多
        class_all = [class1, class2, class3, class4, class5]
        index = class_all.index(max(class_all))
        # # 打哈欠的次数超过某个范围 则代表为打哈欠 和 闭眼冲突
        # if class4 >= 5:
        #     index = 3
        print("此视频属于-------->：", classes_dict[index])

        every_write_list = [each_video, classes_dict[index]]
        write_list.append(every_write_list)
        every_write_list = []
    print(write_list)
import pandas as pd

# 存储预测数据
data = pd.DataFrame(write_list)
data.to_csv('output.csv')

