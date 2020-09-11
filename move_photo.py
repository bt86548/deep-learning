import os
from glob import glob
import shutil

base_path = r'E:/google drive/iii_course/HOT/img_distinct/car_raw_data/secondcar_raw'

# Step 1 list all brands
brands_list = os.listdir(base_path)

def second_layer(brand):
    return os.listdir(f"{base_path}/{brand}")
def third_layer(brand, series):
    return os.listdir(f"{base_path}/{brand}/{series}")
def fourth_layer(brand, series, year):
    return os.listdir(f"{base_path}/{brand}/{series}/{year}")
def fifth_layer(brand, series, year, types):
    return os.listdir(f"{base_path}/{brand}/{series}/{year}/{types}")
i = 0
for brand in brands_list:
    #print(brand)
    serieses = second_layer(brand)
    for series in serieses:
        years = third_layer(brand, series)
        try:
            for year in years:
                types = fourth_layer(brand, series,year)
                # for type_ in types:
                #     images =  fifth_layer(brand, series, year, type_)
                    # for image in images:
                    #     original_dir = f"{base_path}/{brand}/{series}/{year}/{type_}/{image}"
                    #     new_dir = f"{base_path}/{brand}/{series}/{image}"
                    #     shutil.move(original_dir, new_dir)
                for image in types:
                    original_dir = f"{base_path}/{brand}/{series}/{year}/{image}"
                    new_dir = f"{base_path}/{brand}/{series}/{image}"
                    shutil.move(original_dir, new_dir)
        except:
            continue
    i+=1
    #if i ==5:
    #    break

#print(third_layer(brands_list[4], second_layer(brands_list[4])[0]))



