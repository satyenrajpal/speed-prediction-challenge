import cv2, os
import sys
from solver import Solver
import argparse
""" 
TODO:
 - 
"""
def str2bool(v):
    return v.lower() in ('true')

def main(config):
    solver = Solver(config)
    hf_factor = solver.run()
    solver.test(hf_factor, config.save_txt)
    cv2.destroyAllWindows()



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--vidpath', type=str, default='data/train.mp4', help='video file path')
    parser.add_argument('--test_vidpath', type=str, default='data/test.mp4', help='test video file path')
    parser.add_argument('--txtfile', type=str, default='data/train.txt', help='txt file path')
    parser.add_argument('--vis', type=str2bool, default=False)
    parser.add_argument('--len_gt', type=int, default=20400, help='length of the ground truth labels')
    parser.add_argument('--save_txt', type=str2bool, default=False, help='Toggle whether to save test predictions in a txt file')
    config = parser.parse_args()

    main(config)

