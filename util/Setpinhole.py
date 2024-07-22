import numpy as np
import cv2

def Setpinhole( m, n, r):
   # m---rows of probe
   # n---cols of probe
   # r---radius of pinhole in probe
   pinhole = np.zeros((m, n))
   x = np.array(np.arange(0, m, 1), np.int32)
   y = np.array(np.arange(0, n, 1), np.int32)
   j, i =  np.meshgrid(x, y)
   i = np.reshape(i, -1)
   j = np.reshape(j, -1)
   for p in zip(i,j):

     if (p[0]-m/2)**2 + (p[1]-n/2)**2 <= r**2:

         pinhole[p[0],p[1]]=1

   return pinhole

def Setpinhole2( m, n):
   # m---rows of probe
   # n---cols of probe
   # r---radius of pinhole in probe
   pinhole = np.zeros((m, n))

   for i in range(m):
    for j in range(n):

       a = (i-(m/2))**2
       b = (j-(n/2))**2
       c = np.sqrt(a + b)
       pinhole[i,j] = c / np.sqrt((n/2)**2+(m/2)**2)



   return pinhole


if __name__ == '__main__':
    a = Setpinhole2(512,512)
    cv2.imshow("a",a)
    cv2.waitKey(0)
    print('main prog')