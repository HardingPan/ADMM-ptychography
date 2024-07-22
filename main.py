import sys
sys.path.append("Algorithms")
import argparse
import time

from Algorithms.admm import Admm

if __name__ == '__main__':
    """
    argument init
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions_path', default="../data/positions_test.npy")
    parser.add_argument('--diffset_path', default="../data/diffset_test.npy")
    parser.add_argument('--backnoise_path', default="../data/backnoise.tiff")
    args = parser.parse_args()

    admm = Admm(args)
    
    time_start=time.perf_counter()
    admm.compute()