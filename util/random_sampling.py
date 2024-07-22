import numpy as np


###从一系列的位置和对应的衍射强度图中随机的选取num_samples个
def random_sampling(positions, diffsets, num_samples=64):

    # 1. 随机选择50个位置对应的值
    positions = np.array(positions)
    diffsets = np.array(diffsets)
    selected_indices = np.random.choice(len(positions), num_samples, replace=False)
    # 2. 根据选定的位置，从 diffset 中提取对应的值
    selected_diffset_values = diffsets[selected_indices]
    # 3. 构建新数组 positions1 和 diffset1
    positions1 = positions[selected_indices]
    diffset1 = selected_diffset_values

    return positions1, diffset1
