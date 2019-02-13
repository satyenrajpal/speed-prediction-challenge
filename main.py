import cv2, os
import sys
from solver import Solver
import argparse
""" 
TODO:
 - write eval function - Done
 - test function to spit out test.txt 
 - Add comments to the code - Done
"""
def main(config):
    solver = Solver(config)
    solver.run()


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--vidpath', type=str, default='data/train.mp4', help='video file path')
    parser.add_argument('--test_vidpath', type=str, default='data/test.mp4', help='test video file path')
    parser.add_argument('--txtfile', type=str, default='data/train.txt', help='txt file path')
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--len_gt', type=int, default=20000, help='length of the ground truth labels')
    config = parser.parse_args()

    main(config)

