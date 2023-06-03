# This code is taken from the GitHub repository: https://github.com/praneethchandraa/Data-Ordering-Adversarial-Attacks

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Sized, Iterator
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Lenet5(nn.Module):
    
    def __init__(self)->None:
        super(Lenet5, self).__init__()
        
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
        
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
        
        self.fc1 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(400, 84)),
            ('relu4', nn.ReLU())
        ]))
        
        
        self.fc2 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
        ]))

        
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

surrogate = Lenet5().to(device)

class BatchReorderSampler(Sampler[int]):

    def __init__(self, data_source: Sized, atk_type="lowtohigh", surrogate= surrogate, batch_size=32) -> None:
        self.data_source = data_source
        self.surrogate = surrogate

        self.epoch1 = True
        self.backdoor = False
        
        self.batchOrder = torch.randperm((len(data_source)//batch_size)*batch_size)
        temp = np.array(list(range(self.batchOrder.shape[0])))
        self.batchOrder = torch.from_numpy(temp)
        self.batchOrder = self.batchOrder.reshape(-1, batch_size)    
        print("batch = ", self.batchOrder[:5])    
        
        data = [self.data_source.__getitem__(j) for j in self.batchOrder.view(-1)]
        data, labels = zip(*data)
        self.data = torch.stack(data).to(device)
        self.labels = torch.LongTensor(labels).to(device)
        self.atk_type = atk_type
     
    def __getSurrogateloss__(self, batch):
        
        with torch.no_grad():
            loss = F.nll_loss(self.surrogate(self.data[batch]) ,self.labels[batch])
        return loss.cpu().item()
    
    
    def __iter__(self) -> Iterator[int]:
        
        
        if self.epoch1 == True:
            print('Waiting to Attack')
            for i in range(self.batchOrder.shape[0]):
                yield iter(self.batchOrder[i])
            self.epoch1 = False
        else:

            print('Attacking')
            print("batch order: ", self.batchOrder)
            losses = torch.Tensor([self.__getSurrogateloss__(batch) for batch in self.batchOrder])

            if self.atk_type == "lowtohigh":
              for i in torch.argsort(losses):
                yield iter(self.batchOrder[i])

            elif self.atk_type == "hightolow":
              for i in torch.argsort(losses, descending=True):
                  yield iter(self.batchOrder[i])

            elif self.atk_type == "oscillating in":
              asc_sort_losses = torch.argsort(losses).tolist()
              oscillate_inward_sort = []
              asc_n = len(asc_sort_losses) - 1
              for loss_ind in range(asc_n//2 + 1):
                if loss_ind%2:
                  oscillate_inward_sort.append(asc_sort_losses[loss_ind-1])
                  oscillate_inward_sort.append(asc_sort_losses[loss_ind])
                else:
                  oscillate_inward_sort.append(asc_sort_losses[asc_n - loss_ind])
                  oscillate_inward_sort.append(asc_sort_losses[asc_n - loss_ind-1])
              oscillate_inward_sort_tensor = torch.tensor(np.array(oscillate_inward_sort))
              for i in oscillate_inward_sort_tensor:
                   yield iter(self.batchOrder[i])

            elif self.atk_type == "oscillating out":
              asc_sort_losses = torch.argsort(losses).tolist()
              oscillate_outward_sort = []
              asc_n = len(asc_sort_losses) - 1
              for loss_ind in range(asc_n//2 + 1):
                if loss_ind%2:
                  oscillate_outward_sort.append(asc_sort_losses[loss_ind-1])
                  oscillate_outward_sort.append(asc_sort_losses[loss_ind])
                else:
                  oscillate_outward_sort.append(asc_sort_losses[asc_n - loss_ind])
                  oscillate_outward_sort.append(asc_sort_losses[asc_n - loss_ind-1])
              oscillate_outward_sort_tensor = torch.tensor(np.array(oscillate_outward_sort))
              oscillate_outward_sort_tensor = torch.flip(oscillate_outward_sort_tensor, [0])
              for i in oscillate_outward_sort_tensor:
                  yield iter(self.batchOrder[i])
            
    def __len__(self) -> int:
        return self.batchOrder.shape[0]
    

class BatchShuffleSampler(Sampler[int]):

    def __init__(self, data_source: Sized,  atk_type="lowtohigh", surrogate=surrogate, batch_size=32, backdoor = False) -> None:
        self.data_source = data_source
        
        self.surrogate = surrogate
        self.surrogate_bd = Lenet5().to(device)
        self.batch_size = batch_size

        self.epoch1 = True
        self.batchOrder = torch.randperm((len(data_source)//batch_size)*batch_size)
        temp = np.array(list(range(self.batchOrder.shape[0])))
        self.batchOrder = torch.from_numpy(temp)
        # print("batch order shape = ", self.batchOrder.shape)
        # print("data source length = ", len(data_source))
        # print("batch order = ", self.batchOrder[:10])
        
        data = [self.data_source.__getitem__(j) for j in self.batchOrder.view(-1)]
        data, labels = zip(*data)
        self.data = torch.stack(data).to(device)
        self.data_perturbed = torch.stack(data).to(device)
        print("data shape = ",self.data.shape)
        self.labels = torch.LongTensor(labels).to(device)

        self.atk_type = atk_type

     
    def __getSurrogateloss__(self, batch):
        
        with torch.no_grad():

            loss = F.nll_loss(self.surrogate(self.data[batch: batch+1])
            ,self.labels[batch: batch+1])

        return loss.cpu().item()
    
    
    def __iter__(self) -> Iterator[int]:
        
        if self.epoch1 == True:
            print('Waiting to Attack')
            # print(self.batchOrder.view(-1, self.batch_size)[:2])
            for i in self.batchOrder.view(-1, self.batch_size):
                yield iter(i)
            self.epoch1 = False
            
        else:
            print('Attacking')
            # print("batch order: ",self.batchOrder)

            losses = torch.Tensor([self.__getSurrogateloss__(batch) for batch in self.batchOrder])
            # print("printing losses now _______")
            # print(losses[:5])
            # print(torch.argsort(losses)[:5])
            # print(losses[torch.argsort(losses)[:5]])
            # print("printing losses done _______")
            
            if self.atk_type == "lowtohigh":
              # print("low to high running")
              # print(self.batchOrder[torch.argsort(losses)].view(-1, self.batch_size)[:2])
              for i in self.batchOrder[torch.argsort(losses)].view(-1, self.batch_size):
                  yield iter(i)

            elif self.atk_type == "hightolow":
              print("high to low running")
              # print(self.batchOrder.shape)
              # print(self.batchOrder[torch.argsort(losses, descending=True)].view(-1, self.batch_size).shape)
              for i in self.batchOrder[torch.argsort(losses, descending=True)].view(-1, self.batch_size):
                  yield iter(i)

            elif self.atk_type == "oscillating in":
              print("oscillating in running")
              
              asc_sort_losses = torch.argsort(losses).tolist()

              oscillate_inward_sort = []
              asc_n = len(asc_sort_losses) - 1
              for loss_ind in range(asc_n//2 + 1):
                oscillate_inward_sort.append(asc_sort_losses[asc_n - loss_ind])
                oscillate_inward_sort.append(asc_sort_losses[loss_ind])
              oscillate_inward_sort_tensor = torch.tensor(np.array(oscillate_inward_sort))

              for i in self.batchOrder[oscillate_inward_sort_tensor].view(-1, self.batch_size):
                  yield iter(i)
            
            elif self.atk_type == "oscillating out":
              print("oscillating out running")
              asc_sort_losses = torch.argsort(losses).tolist()
              oscillate_outward_sort = []
              asc_n = len(asc_sort_losses) - 1
              for loss_ind in range(asc_n//2 + 1):
                oscillate_outward_sort.append(asc_sort_losses[asc_n - loss_ind])
                oscillate_outward_sort.append(asc_sort_losses[loss_ind])
              oscillate_outward_sort_tensor = torch.tensor(np.array(oscillate_outward_sort))
              oscillate_outward_sort_tensor = torch.flip(oscillate_outward_sort_tensor, [0])
              for i in self.batchOrder[oscillate_outward_sort_tensor].view(-1, self.batch_size):
                  yield iter(i)
            
            print('Attack successful')
        

    def __len__(self) -> int:
        return self.batchOrder.view(-1, self.batch_size).shape[0]