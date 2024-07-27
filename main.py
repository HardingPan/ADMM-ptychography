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
    parser.add_argument('--positions_path', default="../data/data_test/p4.npy")
    parser.add_argument('--diffset_path', default="../data/data_test/d4.npy")
    parser.add_argument('--backnoise_path', default="../data/backnoise.tiff")
    parser.add_argument('--save_path', default="./res/res.png")
    args = parser.parse_args()

    admm = Admm(args)
    
    time_start=time.perf_counter()
    target, target_ph, probe_re, err = admm.compute('T_ADMM') # 重建
    time_end =time.perf_counter()
    time_consume=time_end-time_start
    print("time consume ", time_consume)

    plot_and_save_results(target, target_ph, probe_re, args)

    # admm
    # 0.17298467470286424 1
    # End of iterations
    # time consume  301.44257737501175

    # ladmm
    # 0.15938004026112076 1
    # End of iterations
    # time consume  260.55320445800317

    # tadmm
    # 0.1610870967135734 1
    # End of iterations
    # time consume  255.66520066598605


