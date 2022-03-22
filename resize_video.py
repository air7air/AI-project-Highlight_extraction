# Import everything needed to edit video clips 
from moviepy.editor import *
import os
import glob

def make_resize_video(video_name):
    # loading video dsa gfg intro video and getting only first 5 seconds 
    # fname = 'first_test.mp4'
    fname = video_name
    clip1 = VideoFileClip('./data_lol_full/raw/'+fname)

    # 원본 video width height 의심되면 아래 주석 지우고 확인 ~
    # # getting width and height of clip 1 
    # w1 = clip1.w 
    # h1 = clip1.h 
    
    # print("Width x Height of clip 1 : ", end = " ") 
    # print(str(w1) + " x ", str(h1)) 
    
    # print("---------------------------------------") 
    
    # # resizing video downsize 128,128 
    clip2 = clip1.resize((128,128)) 
    
    # resize video width height 의심되면 아래 주석 지우고 확인 ~
    # # getting width and height of clip 1 
    # w2 = clip2.w 
    # h2 = clip2.h 
    
    # print("Width x Height of clip 2 : ", end = " ") 
    # print(str(w2) + " x ", str(h2)) 
    
    # print("---------------------------------------") 
    
    # resize 한 영상 넣을 directory생성 만들었으면 주석처리할 것
    # resize 한 영상 넣을 directory생성 안 만들었으면 주석해제할 것
    # os.mkdir('./resize')
    # clip2.ipython_display()
    clip2.write_videofile('./data_lol_full_resize/raw/'+fname)

#20200301_LIYAB_HKATEMP_MPY_wvf_snd.mp3

    
# make_resize_video()
video_list = sorted(glob.glob('./data_lol_full/raw/*.mp4'))
# print(video_list)

for i in video_list:
    title, name = os.path.normpath(i).split(os.sep)[-2:]
    # filename만 받기
    # name = os.path.splitext(name)[0]
    # 나는 그냥 .mp4까지 받음
    # print('name {}'.format(name))
    make_resize_video(name)