import numpy as np
import cv2
def diff_xy( image ):

    n1, n2 = image.shape

    diff_image = np.zeros(shape= (n1,n2,2), dtype=np.complex64)

    diff_image[0:n1-1, :, 0]  = np.diff(image, n=1, axis=0) #dy
    diff_image[:, 0:n2-1, 1]  = np.diff(image, n=1, axis=1) #dx

    return diff_image








def diff_xy_transpose(image):

     n1 ,n2 ,n3 = image.shape
     u1 = np.zeros(shape=(n1 , n2) , dtype=np.complex64)
     u2 = np.zeros(shape=(n1, n2), dtype=np.complex64)


     c1 = circshift(image[:,:,0], 1, 0)
     u1 =c1-image[:,:,0]
     u1[n1-1, :] = c1[n1-1, :]
     u1[0, :] = -c1[1,:]

     c2 = circshift(image[:,:,1], 0, 1)
     u2 = c2 - image[:, :, 1]
     u2[:, n2-1] = c2[:, n2-1]
     u1[:, 1] = -c2[:, 1]

     u=u1+u2

     return u

##   u代表原矩阵，shiftnum1代表行，shiftnum2代表列。
def circshift(u,shiftnum1,shiftnum2):
    h,w = u.shape
    if shiftnum1 < 0:
        u = np.vstack((u[-shiftnum1:,:],u[:-shiftnum1,:]))
    else:
        u = np.vstack((u[(h-shiftnum1):, :], u[:(h-shiftnum1),:]))
    if shiftnum2 > 0:
        u = np.hstack((u[:, (w - shiftnum2):], u[:, :(w - shiftnum2)]))
    else:
        u = np.hstack((u[:,-shiftnum2:],u[:,:-shiftnum2]))

    return u


def sign(image):


    out=np.where(abs(image) > 0, np.exp(1j*np.angle(image)) ,0)

    return out

if __name__ =='__main__':

    img = cv2.imread("E:/pythonProject/pie-penalize/image_data/Set12/cameraman256.png")
    f_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

    out = diff_xy(f_gray)
    out = diff_xy_transpose(out)

    cv2.imshow("resized", out)

    cv2.waitKey(0)

