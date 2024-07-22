import numpy as np


def phase_unwrap(image1):
    pi = 3.141592653589793
    [row, col] = image1.shape
    image = np.angle(image1)
    ans = image
    for i in range(row):
       for j in range(1, col):
           delta = image[i][j] - image[i][j-1]
           if delta > pi:
               # while(- pi < delta < pi):
                 delta -= 2 * pi

           elif delta < -pi:
               # while(- pi < delta < pi):
                delta += 2 * pi

           ans[i][j] = ans[i][j-1] + delta


    return abs(image1)*np.exp(1j * ans)