#! encoding: UTF-8

import os
import shutil
import cv2



def video2img(videos_src_path, videos_save_path):
    '''


    :param videos_src_path: 视频存放文件的上一级文件夹
    :param videos_save_path: 图片保存的上一级文件夹
    例如：
        Test/A/1.mp4
        Test/B/2.mp4
    则：
        videos_src_path = 'Test/'
        新建同级文件夹 'Test_out'
        videos_save_path = 'Test_out/'

    :return:
    '''
    frame_count = 1
    classname = []
    img_path = []
    for root, _, videos in os.walk(videos_src_path):
        # 遍历文件建立文件夹
        for _ in videos:
            # 获得视频存储文件夹名字
            folder_name = root.split('/')[-1]
            each_video_save_folder = videos_save_path + '/' + str(folder_name)
            # 创建输出帧文件夹
            if os.path.exists(each_video_save_folder):
                shutil.rmtree(each_video_save_folder)  # 递归删除文件夹(非空文件夹删除)
                os.mkdir(each_video_save_folder)
            else:
                os.mkdir(each_video_save_folder)

        for each_video in videos:
            # 获得视频存储文件夹名字
            folder_name = root.split('/')[-1]
            each_video_name = each_video.split('.')[0]
            each_video_save_folder = videos_save_path + '/' + str(folder_name)
            # 得到每个视频文件路径
            each_video_full_path = os.path.join(root, each_video)
            print("当前解帧视频的完全路径each_video_full_path-----> ", each_video_full_path)
            each_video_save_full_path = os.path.join(each_video_save_folder) + '/'
            print("视频解帧提取地址each_video_save_full_path------->", each_video_save_full_path)
            print("当前读入视频的名字each_video_name", each_video_name)

            cap = cv2.VideoCapture(each_video_full_path)
            success = True
            while(success):
                success, frame = cap.read()
        #        print('Read a new frame: ', success)
                params = []
                params.append(cv2.IMWRITE_PXM_BINARY)
                params.append(1)
                cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, params)

                classname.append(folder_name)
                img_path.append(each_video_name + "_%d.jpg" % frame_count)
                # print(classname, img_path)
                #


                frame_count = frame_count + 1

            cap.release()
    # 将此帧图像写入到列表
    import pandas as pd

    data = pd.DataFrame({'classname': classname, 'img': img_path})
    data.to_csv("img_list.csv", index=False)
    print("-----------视频逐帧提取完成-----------")


if __name__ == '__main__':

    videos_src_path = './data/Test/'
    videos_save_path = './data/Test_out'
    video2img(videos_src_path, videos_save_path)

