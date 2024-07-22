import numpy as np
import cv2

def C(image , sigma):  #sigma -- sampling rate

    n1, n2 = image.shape
    u = np.zeros((n1,n2), dtype=np.float64)

    for r in range(sigma):
        for c in range(sigma):
            u[::sigma , ::sigma] = u[::sigma , ::sigma] + image[r::sigma, c::sigma]

    out = u[::sigma, ::sigma] / sigma**2

    return out


def CT(image , sigma):

   n1, n2 = image.shape

   u = np.zeros((n1*sigma,n2*sigma),dtype=np.float64)

   for r in range(sigma):
       for c in range(sigma):

         u[r::sigma , c::sigma] = image / sigma**2


   return u



if __name__ == '__main__':


 img =cv2.imread("E:/pythonProject/pie-penalize/image_data/feiyuan.png")
 f_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 out = cv2.pyrDown(f_gray)
 out = cv2.pyrDown(out)
 out = cv2.pyrDown(out)
 c =out[9:73,9:73]

 #out = CT(out,2)

 cv2.imwrite("E:/pythonProject/pie-penalize/image_data/feiyuantanzhen.png",c)