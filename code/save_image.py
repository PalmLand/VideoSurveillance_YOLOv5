'''
import cv2
#该脚本将视频转为图片

# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image,addr,num):
    address = addr + str(num)+ '.jpg'
    cv2.imwrite(address,image)

# 读取视频文件
videoCapture = cv2.VideoCapture("../SLP_demo.mp4")#视频文件路径
# 通过摄像头的方式
# videoCapture=cv2.VideoCapture(1)
  
#读帧
success, frame = videoCapture.read()
print(success)
print(frame)
i = 0
timeF = 12
j=0
while success :
    i = i + 1
    if (i % timeF == 0):
        j = j + 1
        save_image(frame,'../data/my_images/image',j)#图片要保存到的路径
        print('save image:',i)
    success, frame = videoCapture.read()
'''   
    
    

import cv2

vc = cv2.VideoCapture("./yolov5/runs/detect/exp/SLP_demo.mp4")#视频文件路径
if vc.isOpened():
    open, frame = vc.read()
else:
    open = False
    
i = 1    
while open:
    ret, frame = vc.read()
    #print(frame)
    if frame is None:
        break
    if ret == True:
        save_path = './data/frame_'+str(i)+'.jpg'
        cv2.imshow('result',frame)
        cv2.imwrite(save_path,frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    i += 1
    print(i)
print('DONE')
vc.release()
cv2.destroyAllWindows()