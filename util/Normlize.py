import numpy as np



def Normlize(img, a, b):

    min = np.min(img)
    max = np.max(img)
    Norm_img = (img-min) / (max - min ) * (b -a)


    return Norm_img


if __name__ == "__main__":

    img = np.load('E:/pythonProject/pie-penalize/image_data/re_am.npy')
    out = Normlize(img , 0 ,1)
    print(np.min(out),np.max(out))