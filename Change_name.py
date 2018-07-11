import os

def change_name(path):
    '''

    :param path: 视频文件的上级文件夹
    例如：
        ./data/Test/C1/a.mp4
        ./data/Test/C2/b.mp4
    则:
        path = ./data/Test
    :return:
    '''
    for root, _, videos in os.walk(path):
        count = 0
        for each_video in videos:
            # 文件夹名就是类别名

            foldername = root.split('/')[-1]
            # print(each_video)
            Olddir = path + '/' + foldername + '/' + each_video
            print(Olddir)
            each_video_name = each_video.split('.')[0]
            each_video_style = each_video.split('.')[1]
            Newdir = path + '/' + foldername + '/' + foldername + '_' + str(count) + '.' + each_video_style
            print(Newdir)
            os.rename(Olddir, Newdir)
            count += 1
    print("--------------更名完毕------------")

if __name__ == '__main__':
    change_name('./data/Test')