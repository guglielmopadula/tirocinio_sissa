import torchvision
import torchvision.transforms as transforms
import torch

import numpy as np
from torch.utils.data import random_split

from matplotlib import pyplot as plt

import os
import time


if __name__ == '__main__':



    dataset=torch.utils.data.TensorDataset(torch.rand(600,245,3),torch.rand(600,264,2))
    trainset,testset = random_split(dataset, [400,200])    


    max_num_workers = os.cpu_count() #refer to https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py
    
    plt.xlabel('num_workers') 
    plt.ylabel('Total Time(Sec)')
    
    batch_size_list = [20]
    num_workers_list = np.arange(max_num_workers + 1)
    for batch_size in batch_size_list:
        total_time_per_num_workers = []
        for num_workers in num_workers_list:
            loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=num_workers,
                                                pin_memory=True)
            
            t1 = time.time()
            for _ in loader: pass
            t2 =time.time()
            
            total_time = t2 - t1
            total_time_per_num_workers.append(total_time)
            print(f"batch_size{batch_size}, num_workers{num_workers}, total_time(sec): ", total_time)
        plt.plot(num_workers_list, total_time_per_num_workers)
    plt.legend([f"batch size {batch_size}" for batch_size in batch_size_list])
    plt.show()
    