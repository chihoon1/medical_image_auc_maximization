import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--test_batchsize', type=int, default=128)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--data', type=str, default="breastmnist") 
parser.add_argument('--server', type=str, default="faster")
parser.add_argument('--pos_class', type=int, default=0)
parser.add_argument('--task_index', type=int, default=0)

args = parser.parse_args()
