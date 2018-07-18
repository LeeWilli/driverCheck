import os, random, shutil, sys
 
 
def copyFile(fileDir):

	# 0 计算出来文件夹中文件数量

	ls = os.listdir(fileDir)
	count = 0
	for i in ls:
	    if os.path.isfile(os.path.join(fileDir,i)):
	        count += 1
	print (count)
	pick_num = int(count * 0.2) 
    # 1
	pathDir = os.listdir(fileDir)
 
    # 2
	sample = random.sample(pathDir, pick_num)
	print (sample)
	
	# 3
	for name in sample:
		# 如果是copy的话，使用copyfile
		shutil.move(fileDir+name, tarDir+name)
if __name__ == '__main__':

	# num = (sys.argv[1])
	fileDir = (sys.argv[1])
	tarDir = (sys.argv[2])
	copyFile(fileDir)
