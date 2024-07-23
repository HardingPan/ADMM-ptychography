import sys
sys.path.append("Algorithms")
import argparse
import time

from Algorithms.admm import *

if __name__ == '__main__':
    """
    argument init
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions_path', default="../data/positions_test.npy")
    parser.add_argument('--diffset_path', default="../data/diffset_test.npy")
    parser.add_argument('--backnoise_path', default="../data/backnoise.tiff")
    parser.add_argument('--save_path', default="./res/res.png")
    args = parser.parse_args()

    admm = Admm(args)
    
    time_start=time.perf_counter()
    target, target_ph, probe_re = admm.compute('various_PIE') # 重建
    time_end =time.perf_counter()
    time_consume=time_end-time_start
    print("time consume ", time_consume)

    plot_and_save_results(target, target_ph, probe_re, args)

