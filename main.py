import torch
import torchvision
import torchvision.transforms as transforms


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import matplotlib.pyplot as plt

import json
from train import train
from test import test

import os
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_num=100
#use ImageNet-2012 dataset
# trainset = torchvision.datasets.ImageFolder('./train', transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_num, shuffle=True)
testset = torchvision.datasets.ImageFolder('./val', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_num, shuffle=False)

idx2label = []
cls2label = {}
with open("./imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch', default=100, type=int,
                        help='number of each process batch number')
    args = parser.parse_args()
    ###########################################################
    args.world_size = args.gpus * args.nodes                  #
    os.environ['MASTER_ADDR'] = '127.0.0.1'                   #
    os.environ['MASTER_PORT'] = '8888'                        #
    mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)#
    ###########################################################

if __name__ == "__main__":
    main()
    test(testloader, idx2label)
