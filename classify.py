import os
import pandas
import shutil
from glob import glob
from tqdm import tqdm
import statistics
import numpy
import shutil

#把小於100張的train都刪掉
path = r'E:\google drive\iii_course\HOT\img_distinct\classify\test_classify'
x  = os.listdir(path)

for i in x:
    # print('{}:'.format(i),'長度:{0}'.format(len(os.listdir(path+'\\'+i))))
    if len(os.listdir(path+'\\'+i)) < 500:
        remove_path = os.path.join(path,i)
        # print(remove_path)
        shutil.rmtree(remove_path)
        # print(i)

# #分類車型
# x  = os.listdir('C:/Users/Big data/Desktop/outside')
# # # folder_locate = os.listdir('E:/google drive/iii_course/HOT/img_distinct/classify/classify_car')
# #車型目標位置
# path = r'E:\google drive\iii_course\HOT\img_distinct\classify\classify_car\Toyota'
# #轉移車型位置
# path_photo = os.listdir(path)
# search_string = 'Toyota_corona'
# for i in path_photo:
#     if search_string in i[:15] or search_string.upper() in i[:15] or search_string.lower() in i[:15] or search_string.title() in i[:15] and (search_string != i) :
#         # print(glob(path+'\\'+i+'\*.jpg'))
#         old_dir = glob(path+'\\'+i+'\*.jpg')
#         new_dir = f'{path}\{search_string}\\'
#         print(i)
#         for j in tqdm(old_dir):
#             shutil.move(j, new_dir+j.split('\\')[-1])
#         remove_path = os.path.join(path,i)
#         if len(os.listdir(remove_path)) == 0:
#             os.rmdir(remove_path)





#物件導向
# def move(photo):
#     a = photo.split('_')
#     brand_locate = f'E:/google drive/iii_course/HOT/img_distinct/classify/classify_car/{a[0]}/{a[1]}/{photo}'
#     return brand_locate

# for i in tqdm(range(len(x))):
#     try:
#         new_dir = move(x[i])
#         old_dir = f'C:/Users/Big data/Desktop/outside/{x[i]}'
#         shutil.move(old_dir, new_dir)
#     except:
#         continue




#移動照片
# x  = os.listdir('C:/Users/Big data/Desktop/outside')
# folder_locate = os.listdir('E:/google drive/iii_course/HOT/img_distinct/classify/classify_car')

# def move(photo):
#     a = photo.split('_')
#     brand_locate = f'E:/google drive/iii_course/HOT/img_distinct/classify/classify_car/{a[0]}/{a[1]}/{photo}'
#     return brand_locate

# for i in tqdm(range(len(x))):
#     try:
#         new_dir = move(x[i])
#         old_dir = f'C:/Users/Big data/Desktop/outside/{x[i]}'
#         shutil.move(old_dir, new_dir)
#     except:
#         continue





