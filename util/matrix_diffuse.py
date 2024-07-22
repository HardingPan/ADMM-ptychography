import numpy as np


def create_mask_matrix(n, m, r):
    # Creating an n*m matrix of zeros
    matrix = np.zeros((n, m))

    # Center of the matrix
    center_i, center_j = (n - 1) / 2.0, (m - 1) / 2.0

    # Calculate maximum distance from the center to any edge
    max_dist = max(center_i, center_j)

    # Computing the distance of each element from the center and normalizing it
    # for i in range(n):
    #     for j in range(m):
    #         dist = max(abs(i - center_i), abs(j - center_j))
    #         matrix[i, j] = dist / max_dist
    # matrix = np.zeros((n, m))

    # 计算圆心位置
    center_x, center_y = n // 2, m // 2

    # 遍历矩阵的每个元素
    for i in range(n):
        for j in range(m):
            # 计算当前点到圆心的距离
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if distance < r:
                # 根据距离设置灰度值，使得灰度值在圆内随半径增大而增大
                matrix[i, j] = (1 - (distance / r))

    return matrix


