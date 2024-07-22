import os
import cv2
import pandas as pd
import  numpy as np

def getphotos(folder_path):
    path_list = []
    photos = []
    filenames = os.listdir(folder_path)
    # filenames = sorted(filenames, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    for filename in filenames:
        a = os.path.join(folder_path, filename)
        path_list.append(a)
        image = cv2.imread(a,-1)
        if image.ndim == 3:
         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        photos.append(np.array(image,dtype=float ))

    return photos


def getexcel(folder_path):
    data = []
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        # 读取Excel文件
        df = pd.read_excel(io=file_path, header=3, sheet_name='Sheet1')
        numpy_array = df.values
        data.append(numpy_array)
    return data


if __name__ == "__main__":
    df = getphotos('C:/Users/86364/Desktop/yanshetuxixang/yanshetuxixang')
    # df = pd.read_excel(io='C:/Users/86364/Desktop/yanshetuxixang/yanshetuxixang/WinCamExcelData_11_28_2023_17_29_6.xlsx', header= 3) #从第4行开始读取excel文件
    # numpy_array = df.values
    print(df)
